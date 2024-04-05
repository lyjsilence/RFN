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
from models import RFN
from Baselines import GRU_ODE, ODELSTM, ODERNN, GRU_D
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


def process():
    path = r'data\full_non_padding.csv'
    data = pd.read_csv(path)
    select_col = ['BIIB', 'BMRN', 'CELG', 'REGN', 'VRTX', 'GILD', 'INCY', 'MYL']
    data = data[select_col]

    fig, axes = plt.subplots(len(select_col), 1, sharex=True, figsize=(6, 6))

    # 绘制每个子图
    for i, col in enumerate(select_col):
        axes[i].scatter(range(len(data)), data[col], label=col, s=1)
        axes[i].legend(loc=3)
    axes[-1].set_xlabel('time')
    axes[-1].set_xticks(np.arange(0, 74500, 20000))  # 设置刻度位置
    axes[-1].set_xticklabels(['26/Jul/2016', '7/Oct/2016', '21/Dec/2016', '6/Mar/2017'])
    axes[-1].set_xlabel('time', fontsize=16)
    plt.savefig('NASDAQ.png', dpi=330)

    plt.show()


    mask = 1 - pd.isnull(data).astype(int)
    data = pd.concat([data, mask], axis=1)
    columns = [f'ts_{str(i)}' for i in range(len(select_col))] + [f'mask_{str(i)}' for i in range(len(select_col))]
    data.columns = columns
    data = data.fillna(0)
    data[[f'ts_{str(i)}' for i in range(len(select_col))]] = data[[f'ts_{str(i)}' for i in range(len(select_col))]] / 100

    num_blocks = data.shape[0] // 75
    data = data.iloc[:num_blocks * 75]
    data['idx'] = np.repeat(np.arange(num_blocks) + 1, 75)
    data['time'] = np.tile(np.arange(1, 76), num_blocks)
    data['time'] = data['time'] / 10
    data = data[['idx', 'time'] + columns]
    return data


parser = argparse.ArgumentParser(description="RFN for NASDAQ process datasets")
parser.add_argument('--seed', type=int, default=0, help='The random seed')
parser.add_argument('--save_dirs', type=str, default='results/NASDAQ', help='The dirs for saving results')
parser.add_argument('--log', type=bool, default=True, help='log the training process')
parser.add_argument('--cuda', type=int, default=0)

# settings for experiments
parser.add_argument('--type', type=str, default='async', choices=['sync', 'async'])
parser.add_argument('--model_name', type=str, default='RFN', help='The model want to implement',
                    choices=['GRUODE', 'ODELSTM', 'ODERNN', 'GRU-D', 'RFN'])
parser.add_argument('--start_exp', type=int, default=0, help='The number of experiment')
parser.add_argument('--num_exp', type=int, default=5, help='The number of experiment')
parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--dt', type=float, default=0.1)
parser.add_argument('--val_start', type=int, default=150)
parser.add_argument('--val_freq', type=int, default=5)

# settings for sequential model
parser.add_argument('--input_dim', type=int, default=8, help='The input dimension of time series')
parser.add_argument('--memory_dim', type=int, default=10, help='The memory dimensions for one vari12able in marginal block')
parser.add_argument('--marginal', type=str, default='GRUODE', help='The models for the marginal block')
parser.add_argument('--atol', type=float, default=1e-4)
parser.add_argument('--rtol', type=float, default=1e-2)

# settings for flow model
parser.add_argument('--solver', type=str, default='dopri5')
parser.add_argument('--num_blocks', type=int, default=1, help='Number of CNF blocks in Flow')
parser.add_argument('--hidden_dim', type=int, default=128, help='The hidden dimension of linear layers')
parser.add_argument('--act_fn', type=str, default='softplus', choices=['tanh', 'softplus', 'elu'])
parser.add_argument('--rademacher', type=bool, default=False)
parser.add_argument('--flow_time', type=float, default=1)
parser.add_argument('--train_T', type=bool, default=True)
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

    data = process()

    T = np.max(data['time'])
    # train-test-split
    data_idx = np.unique(np.array((data['idx'])))
    train_idx, val_idx = train_test_split(data_idx, train_size=0.7, random_state=42)
    val_idx, test_idx = train_test_split(val_idx, train_size=0.5, random_state=42)

    train_data = data[data['idx'].isin(train_idx)]
    val_data = data[data['idx'].isin(val_idx)]
    test_data = data[data['idx'].isin(test_idx)]

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
    for exp_id in range(args.start_exp, args.num_exp):
        print(f'Training models={args.model_name}; Exp_id={exp_id}')
        print(f'HyperParams: lr={args.lr}, batch_size={args.batch_size}, memory_size={args.input_dim*args.memory_dim}, '
              f'num_CNF={args.num_blocks}, dropout={args.dropout}')

        if args.model_name == 'RFN':
            model = RFN(args, device=device).to(device)
        elif args.model_name == 'GRUODE':
            model = GRU_ODE(args, device=device).to(device)
        elif args.model_name == 'ODELSTM':
            model = ODELSTM(args, device=device).to(device)
        elif args.model_name == 'ODERNN':
            model = ODERNN(args, device=device).to(device)
        elif args.model_name == 'GRU-D':
            model = GRU_D(args, device=device).to(device)

        Trainer(model, dl_train, dl_val, dl_test, args, device, exp_id, epoch_max=200)
