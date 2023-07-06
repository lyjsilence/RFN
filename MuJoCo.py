import os
import numpy as np
import torch
import utils
import copy
import pandas as pd
import argparse
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models import RFN_sync, RFN_async
from Baselines import GRU_ODE, ODELSTM, ODERNN, GRU_D, GRU_delta_t
from models_training import Trainer
import utils
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

'''
The data generation and data preprodcessing is modified from 
https://github.com/YuliaRubanova/latent_ode/blob/master/mujoco_physics.py
'''
class HopperPhysics(object):
    t = 150
    D = 14

    full_data_file = 'full_data.csv'
    async_data_file = 'async_sample_data.csv'
    sync_data_file = 'sync_sample_data.csv'

    def __init__(self, root, n_training_samples, args, normalize=False):
        self.root = root
        self.args = args

        # generate data if data is not exists
        if not self._check_exists():
            self._generate_dataset(n_samples=n_training_samples)
            self._save_dataset()

        self.full = pd.read_csv(os.path.join(self.data_folder, self.full_data_file))
        self.sample_sync = pd.read_csv(os.path.join(self.data_folder, self.sync_data_file))
        self.sample_async = pd.read_csv(os.path.join(self.data_folder, self.async_data_file))

        if normalize:
            self.full = self.normalize(self.full, 'full')
            self.sample_sync = self.normalize(self.sample_sync, 'sync')
            self.sample_async = self.normalize(self.sample_async, 'async')


    def visualize(self, traj, plot_name='traj', dirname='hopper_imgs', video_name=None):
        r"""Generates images of the trajectory and stores them as <dirname>/traj<index>-<t>.jpg"""

        T, D = traj.size()

        traj = traj.cpu() * self.data_max.cpu() + self.data_min.cpu()

        try:
            from dm_control import suite  # noqa: F401
        except ImportError as e:
            raise Exception('Deepmind Control Suite is required to visualize the dataset.') from e

        try:
            from PIL import Image  # noqa: F401
        except ImportError as e:
            raise Exception('PIL is required to visualize the dataset.') from e

        def save_image(data, filename):
            im = Image.fromarray(data)
            im.save(filename)

        os.makedirs(dirname, exist_ok=True)

        env = suite.load('hopper', 'stand')
        physics = env.physics

        for t in range(T):
            with physics.reset_context():
                physics.data.qpos[:] = traj[t, :D // 2]
                physics.data.qvel[:] = traj[t, D // 2:]
            save_image(
                physics.render(height=480, width=640, camera_id=0),
                os.path.join(dirname, plot_name + '-{:03d}.jpg'.format(t))
            )

    def _generate_dataset(self, n_samples):
        print('Generating dataset...')

        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(123)

        self.full = pd.DataFrame()
        self.sample_sync = pd.DataFrame()
        self.sample_async = pd.DataFrame()

        for i in range(n_samples):
            idx = np.array([i] * self.t)
            time = np.linspace(0, 1.49, self.t)
            if len(idx.shape) == 1:
                idx = np.expand_dims(idx, axis=1)
            if len(time.shape) == 1:
                time = np.expand_dims(time, axis=1)

            sample = self._generate_random_trajectories()
            sample_sync = self._sampling_sync(sample)
            sample_async = self._sampling_async(sample)
            sample = np.concatenate([idx, time, sample], axis=1)
            sample_async = np.concatenate([idx, time, sample_async], axis=1)
            sample_sync = np.concatenate([idx, time, sample_sync], axis=1)

            # delete time point when there is no observations
            delete_idx_async = np.where(np.sum(sample_async[:, 2: 16], axis=1) == 0)
            delete_idx_sync = np.where(np.sum(sample_sync[:, 2: 16], axis=1) == 0)
            sample_async = np.delete(sample_async, delete_idx_async, axis=0)
            sample_sync = np.delete(sample_sync, delete_idx_sync, axis=0)

            self.full = pd.concat([self.full, pd.DataFrame(sample)], axis=0)
            self.sample_sync = pd.concat([self.sample_sync, pd.DataFrame(sample_sync)], axis=0)
            self.sample_async = pd.concat([self.sample_async, pd.DataFrame(sample_async)], axis=0)

        # Restore RNG.
        np.random.set_state(st0)
        self.full.columns = ['idx', 'time'] + ['ts_' + str(i) for i in range(self.D)]
        self.sample_async.columns = ['idx', 'time'] + ['ts_' + str(i) for i in range(self.D)] + \
                                    ['mask_' + str(i) for i in range(self.D)]
        self.sample_sync.columns = ['idx', 'time'] + ['ts_' + str(i) for i in range(self.D)] + \
                                   ['mask_' + str(i) for i in range(self.D)]

    def _generate_random_trajectories(self):

        try:
            from dm_control import suite  # noqa: F401
        except ImportError as e:
            raise Exception('Deepmind Control Suite is required to generate the dataset.') from e

        env = suite.load('hopper', 'stand')
        physics = env.physics

        data = np.zeros((self.t, self.D))

        with physics.reset_context():
            # x and z positions of the hopper. We want z > 0 for the hopper to stay above ground.
            physics.data.qpos[:2] = np.random.uniform(0, 0.5, size=2)
            physics.data.qpos[2:] = np.random.uniform(-2, 2, size=physics.data.qpos[2:].shape)
            physics.data.qvel[:] = np.random.uniform(-5, 5, size=physics.data.qvel.shape)
        for t in range(self.t):
            data[t, :self.D // 2] = physics.data.qpos
            data[t, self.D // 2:] = physics.data.qvel
            physics.step()
        return data

    def _sampling_async(self, p):
        p_sample = copy.deepcopy(p)
        if len(p_sample.shape) == 1:
            p_sample = np.expand_dims(p_sample, axis=1)

        mask = np.ones_like(p_sample)
        for c in range(p_sample.shape[1]):
            missing_idx = np.random.choice(self.t, int(self.t * self.args.missing_rate), replace=False)
            p_sample[missing_idx, c] = 0
            mask[missing_idx, c] = 0
        return np.concatenate([p_sample, mask], axis=1)

    def _sampling_sync(self, p):
        p_sample = copy.deepcopy(p)
        if len(p_sample.shape) == 1:
            p_sample = np.expand_dims(p_sample, axis=1)

        mask = np.ones_like(p_sample)
        missing_idx = np.random.choice(self.t, int(self.t * self.args.missing_rate), replace=False)
        p_sample[missing_idx, :] = 0
        mask[missing_idx, :] = 0
        return np.concatenate([p_sample, mask], axis=1)

    def normalize(self, data, type):
        if type == 'full':
            data_copy = data.copy()
            self.data_min = np.min(data.loc[:, [c.startswith("ts") for c in data.columns]], axis=0)
            self.data_max = np.max(data.loc[:, [c.startswith("ts") for c in data.columns]], axis=0)
            data_copy.loc[:, [c.startswith("ts") for c in data.columns]] \
                = (data.loc[:, [c.startswith("ts") for c in data.columns]] - self.data_min) / (self.data_max - self.data_min)
            return data_copy

        if type == 'sync' or type == 'async':
            sample_data = data.copy()
            mask = np.array(data.loc[:, [c.startswith("mask") for c in data.columns]])
            sample_data.loc[:, [c.startswith("ts") for c in data.columns]] \
                = np.array((data.loc[:, [c.startswith("ts") for c in data.columns]] - self.data_min) / (self.data_max - self.data_min)) * mask
            return sample_data

    def _check_exists(self):
        return os.path.exists(os.path.join(self.data_folder, self.full_data_file)) and \
               os.path.exists(os.path.join(self.data_folder, self.sync_data_file)) and \
               os.path.exists(os.path.join(self.data_folder, self.async_data_file))

    @property
    def data_folder(self):
        return os.path.join(self.root, self.__class__.__name__)

    def get_dataset(self, type):
        if type == 'full':
            return self.full
        elif type == 'sync':
            return self.sample_sync
        elif type == 'async':
            return self.sample_async

    def _save_dataset(self):
        os.makedirs(self.data_folder, exist_ok=True)
        self.full.to_csv(os.path.join(self.data_folder, self.full_data_file), index=False)
        self.sample_sync.to_csv(os.path.join(self.data_folder, self.sync_data_file), index=False)
        self.sample_async.to_csv(os.path.join(self.data_folder, self.async_data_file), index=False)

    def __len__(self):
        return len(self.full)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

parser = argparse.ArgumentParser(description="FLOW_GRUODE for MuJoCo datasets")
parser.add_argument('--seed', type=int, default=0, help='The random seed')
parser.add_argument('--save_dirs', type=str, default='results/Hopper', help='The dirs for saving results')
parser.add_argument('--log', type=bool, default=True, help='log the training process')
parser.add_argument('--cuda', type=int, default=1)

# settings for experiments
parser.add_argument('--missing_rate', type=float, default=0.5)
parser.add_argument('--type', type=str, default='async', choices=['sync', 'async'])
parser.add_argument('--model_name', type=str, default='RFN', help='The model want to implement',
                    choices=['GRU-delta-t', 'GRUODE', 'ODELSTM', 'ODERNN', 'GRU-D', 'RFN'])
parser.add_argument('--num_exp', type=int, default=5, help='The number of experiment')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--min_lr', type=float, default=0.01, help='minimum learning rate')
parser.add_argument('--adaptive_lr', type=str, default='cosine')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--val_start', type=int, default=150)
parser.add_argument('--val_freq', type=int, default=5)

# settings for sequential model
parser.add_argument('--input_dim', type=int, default=14, help='The input dimension of time series')
parser.add_argument('--memory_dim', type=int, default=10, help='The memory dimensions for one variable in marginal block')
parser.add_argument('--marginal', type=str, default='GRUODE', help='The models for the marginal block',
                    choices=['GRUODE', 'ODELSTM', 'ODERNN', 'GRU-D'])
parser.add_argument('--train_indep', type=bool, default=True, help='The models for the marginal block')
parser.add_argument('--atol', type=float, default=1e-9)
parser.add_argument('--rtol', type=float, default=1e-7)

# settings for flow model
parser.add_argument('--solver', type=str, default='dopri5')
parser.add_argument('--num_blocks', type=int, default=1, help='Number of CNF blocks in Flow')
parser.add_argument('--hidden_dim', type=int, default=128, help='The hidden dimension of linear layers')
parser.add_argument('--act_fn', type=str, default='softplus', choices=['tanh', 'softplus', 'elu'])
parser.add_argument('--rademacher', type=bool, default=False)
parser.add_argument('--flow_time', type=float, default=1)
parser.add_argument('--train_T', type=bool, default=True)
parser.add_argument('--batch_norm', type=bool, default=True)
parser.add_argument('--viz', type=bool, default=False)
parser.add_argument('--dropout', type=float, default=0)

# settings for flow regularization
parser.add_argument('--kinetic-energy', type=float, default=0.1, help="int_t ||f||_2^2")
parser.add_argument('--jacobian-norm2', type=float, default=0.1, help="int_t ||df/dx||_F^2")
parser.add_argument('--directional-penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")
args = parser.parse_args()

if __name__ == '__main__':
    set_seed(args.seed)
    device = torch.device("cuda:"+str(args.cuda) if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.save_dirs):
        os.makedirs(args.save_dirs)

    hopper = HopperPhysics(root='data/Mujoco', n_training_samples=5000, args=args)
    data = hopper.get_dataset(args.type)

    # train-test-split
    data_idx = np.unique(np.array((data['idx'])))
    train_idx, val_idx = train_test_split(data_idx, train_size=0.7, random_state=42)
    val_idx, test_idx = train_test_split(val_idx, train_size=0.5, random_state=42)

    train_data = data[data['idx'].isin(train_idx)]
    val_data = data[data['idx'].isin(val_idx)]
    test_data = data[data['idx'].isin(test_idx)]

    if args.viz:
        viz_data = test_data
        viz_data_3 = viz_data[viz_data['time'] == 0.3][['ts_'+str(i) for i in range(14)]]
        viz_data_6 = viz_data[viz_data['time'] == 0.6][['ts_'+str(i) for i in range(14)]]
        viz_data_9 = viz_data[viz_data['time'] == 0.9][['ts_'+str(i) for i in range(14)]]
        df_corr_3 = pd.DataFrame(viz_data_3).corr()
        df_corr_6 = pd.DataFrame(viz_data_6).corr()
        df_corr_9 = pd.DataFrame(viz_data_9).corr()

        fig = plt.figure(figsize=[15, 5])
        plt.subplot(1, 3, 1)
        sns.heatmap(df_corr_3, center=0, annot=False, cmap='YlGnBu')
        plt.title('Correlation matrix at time point 0.3')
        plt.subplot(1, 3, 2)
        sns.heatmap(df_corr_6, center=0, annot=False, cmap='YlGnBu')
        plt.title('Correlation matrix at time point 0.6')
        plt.subplot(1, 3, 3)
        sns.heatmap(df_corr_9, center=0, annot=False, cmap='YlGnBu')
        plt.title('Correlation matrix at time point 0.9')
        plt.savefig(os.path.join(args.save_dirs, 'ground.jpg'))

    train_data = utils.sim_dataset(train_data)
    val_data = utils.sim_dataset(val_data)
    test_data = utils.sim_dataset(test_data)

    dl_train = DataLoader(dataset=train_data, collate_fn=utils.sim_collate_fn,
                          shuffle=True, batch_size=args.batch_size, num_workers=1, pin_memory=False)
    dl_val = DataLoader(dataset=val_data, collate_fn=utils.sim_collate_fn,
                        shuffle=False, batch_size=len(val_data), num_workers=1, pin_memory=False)
    dl_test = DataLoader(dataset=test_data, collate_fn=utils.sim_collate_fn,
                         shuffle=False, batch_size=len(test_data), num_workers=1, pin_memory=False)

    # Training
    for exp_id in range(args.num_exp):
        print(f'Training models={args.model_name}; Exp_id={exp_id}; Data Type={args.type}; Missing rate={args.missing_rate}')
        print(f'HyperParams: lr={args.lr}, batch_size={args.batch_size}, memory_size={args.input_dim*args.memory_dim}, '
              f'num_CNF={args.num_blocks}, dropout={args.dropout}')

        if args.model_name == 'RFN':
            if args.type == 'sync':
                model = RFN_sync(args, device=device).to(device)
            else:
                model = RFN_async(args, device=device).to(device)
        elif args.model_name == 'GRUODE':
            model = GRU_ODE(args, device=device).to(device)
        elif args.model_name == 'ODELSTM':
            model = ODELSTM(args, device=device).to(device)
        elif args.model_name == 'ODERNN':
            model = ODERNN(args, device=device).to(device)
        elif args.model_name == 'GRU-D':
            model = GRU_D(args, device=device).to(device)

        Trainer(model, dl_train, dl_val, dl_test, args, device, exp_id)




