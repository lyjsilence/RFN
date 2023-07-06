import os
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models import RFN_sync, RFN_async
from models_training import Trainer
import utils

from Baselines import GRU_ODE, ODELSTM, ODERNN, GRU_D


warnings.filterwarnings('ignore')

'''
Simulate the GBM process
dS_t = \mu S_t dt + \sigma S_t^\Gamma dW_t
'''

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


parser = argparse.ArgumentParser(description="RFN for GBM process datasets")
parser.add_argument('--seed', type=int, default=0, help='The random seed')
parser.add_argument('--data_dirs', type=str, default='data/Simulation', help='The dirs for saving results')
parser.add_argument('--save_dirs', type=str, default='results/Simulation', help='The dirs for saving results')
parser.add_argument('--log', type=bool, default=True, help='log the training process')
parser.add_argument('--cuda', type=int, default=0)

# settings for experiments
parser.add_argument('--missing_rate', type=float, default=0.5)
parser.add_argument('--type', type=str, default='async', choices=['sync', 'async'])
parser.add_argument('--model_name', type=str, default='RFN', help='The model want to implement',
                    choices=['GRUODE', 'ODELSTM', 'ODERNN', 'GRU-D', 'RFN'])
parser.add_argument('--start_exp', type=int, default=0)
parser.add_argument('--num_exp', type=int, default=5, help='The number of experiment')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
parser.add_argument('--min_lr', type=float, default=0.001, help='minimum learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--val_start', type=int, default=150)
parser.add_argument('--val_freq', type=int, default=5)

# settings for sequential model
parser.add_argument('--input_dim', type=int, default=5, help='The input dimension of time series')
parser.add_argument('--memory_dim', type=int, default=10, help='The memory dimensions for one variable in marginal block')
parser.add_argument('--marginal', type=str, default='GRUODE', help='The models for the marginal block',
                    choices=['GRUODE', 'ODELSTM', 'ODERNN', 'GRU-D'])
parser.add_argument('--train_indep', type=bool, default=True, help='Train each dimension independently')
parser.add_argument('--atol', type=float, default=1e-7)
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

    if not os.path.exists(args.data_dirs):
        os.makedirs(args.data_dirs)
    if not os.path.exists(args.save_dirs):
        os.makedirs(args.save_dirs)

    if not os.path.exists(os.path.join(args.data_dirs, args.type+'_sample_data.csv')):
        process = utils.DataGenertor(args, N=1000)
        process.simulate()
        process.plot(num_samples=3)
        process.save(data_dirs=args.data_dirs)

    data = pd.read_csv(os.path.join(args.data_dirs, args.type+'_sample_data.csv'))

    # train_test_split
    data_idx = np.unique(np.array((data['idx'])))

    train_idx, val_idx = train_test_split(data_idx, train_size=0.7, random_state=args.seed)
    val_idx, test_idx = train_test_split(val_idx, train_size=0.5, random_state=args.seed)

    train_data = data[data['idx'].isin(train_idx)]
    val_data = data[data['idx'].isin(val_idx)]
    test_data = data[data['idx'].isin(test_idx)]

    if args.viz:
        viz_data = test_data
        viz_data_3 = viz_data[viz_data['time'] == 0.3][['ts_0', 'ts_1', 'ts_2', 'ts_3', 'ts_4']]
        viz_data_6 = viz_data[viz_data['time'] == 0.6][['ts_0', 'ts_1', 'ts_2', 'ts_3', 'ts_4']]
        viz_data_9 = viz_data[viz_data['time'] == 0.9][['ts_0', 'ts_1', 'ts_2', 'ts_3', 'ts_4']]
        df_corr_3 = pd.DataFrame(viz_data_3).corr()
        df_corr_6 = pd.DataFrame(viz_data_6).corr()
        df_corr_9 = pd.DataFrame(viz_data_9).corr()

        fig = plt.figure(figsize=[15, 5])
        plt.subplot(1, 3, 1)
        sns.heatmap(df_corr_3, center=0, annot=True, cmap='YlGnBu')
        plt.title('Correlation matrix at time point 0.3')
        plt.subplot(1, 3, 2)
        sns.heatmap(df_corr_6, center=0, annot=True, cmap='YlGnBu')
        plt.title('Correlation matrix at time point 0.6')
        plt.subplot(1, 3, 3)
        sns.heatmap(df_corr_9, center=0, annot=True, cmap='YlGnBu')
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

    for exp_id in range(args.start_exp, args.num_exp):
        print(f'Training models={args.model_name}; Exp_id={exp_id}; Data Type={args.type}; Missing rate={args.missing_rate}')
        print(f'HyperParams: lr={args.lr}, batch_size={args.batch_size}, memory_size={args.input_dim*args.memory_dim}, '
              f'num_CNF={args.num_blocks}, dropout={args.dropout}')
        # set_seed(args.seed)
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