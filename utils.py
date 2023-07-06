import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
import copy
import properscoring as ps
import six

class DataGenertor:
    def __init__(self, args, N):
        self.args = args
        self.N = N  # number of samples
        self.num_series = args.input_dim
        # self.S0 = np.random.uniform(low=9, high=10, size=[self.N, 5])
        self.S0 = np.random.uniform(low=10, high=10, size=[self.N, 5])
        mu_down = np.random.uniform(low=-0.2, high=-0.05, size=[self.N, 2])
        mu_up = np.random.uniform(low=0.05, high=0.2, size=[self.N, 3])
        self.mu = np.concatenate([mu_down, mu_up], axis=1)
        sigma_down = np.random.uniform(low=0.15, high=0.3, size=[self.N, 2])
        sigma_up = np.random.uniform(low=0.15, high=0.3, size=[self.N, 3])
        self.sigma = np.concatenate([sigma_down, sigma_up], axis=1)

    def generate(self, n, T, dt):
        self.t = round(T / dt)

        mu, sigma, S0 \
            = np.expand_dims(self.mu[n, :], axis=0), np.expand_dims(self.sigma[n, :], axis=0), np.expand_dims(
            self.S0[n, :], axis=0)

        p = np.zeros([self.t, self.num_series])

        p[0, :] = S0

        for t in range(self.t - 1):
            # rho = 0.99 * np.sin(7/6 * np.pi * t / self.t)
            rho = 0.99 * np.sin(1 / 2 * np.pi * t / self.t)
            corr_mat = [[1, rho, 0, 0, 0],
                        [rho, 1, 0, 0, 0],
                        [0, 0, 1, rho, rho],
                        [0, 0, rho, 1, rho],
                        [0, 0, rho, rho, 1]]
            rho_mat = np.linalg.cholesky(corr_mat)
            eps = np.transpose(np.dot(rho_mat, np.random.normal(size=[self.num_series, 1])))
            p[t + 1] = p[t] + p[t] * mu * dt + sigma * p[t] * eps * dt ** 0.5
        return p

    def simulate(self, T=1, dt=0.01):
        self.t = round(T / dt)  # number of observation periods
        self.T = T
        self.full = pd.DataFrame()
        self.sample_async = pd.DataFrame()
        self.sample_sync = pd.DataFrame()

        for n in range(self.N):
            idx = np.array([n] * self.t)
            time = np.arange(0, T, dt)
            p = self.generate(n, T, dt)

            # generate samples which is async and sync
            p_sample_async = self.sampling_async(p)
            p_sample_sync = self.sampling_sync(p)

            if len(idx.shape) == 1:
                idx = np.expand_dims(idx, axis=1)
            if len(time.shape) == 1:
                time = np.expand_dims(time, axis=1)
            if len(p.shape) == 1:
                p = np.expand_dims(p, axis=1)

            p = np.concatenate([idx, time, p], axis=1)
            p_sample_async = np.concatenate([idx, time, p_sample_async], axis=1)
            p_sample_sync = np.concatenate([idx, time, p_sample_sync], axis=1)

            # delete time point when there is no observations
            delete_idx_async = np.where(np.sum(p_sample_async[:, 7: 12], axis=1) == 0)
            delete_idx_sync = np.where(np.sum(p_sample_sync[:, 7: 12], axis=1) == 0)
            p_sample_async = np.delete(p_sample_async, delete_idx_async, axis=0)
            p_sample_sync = np.delete(p_sample_sync, delete_idx_sync, axis=0)

            self.full = pd.concat([self.full, pd.DataFrame(p)], axis=0)
            self.sample_async = pd.concat([self.sample_async, pd.DataFrame(p_sample_async)], axis=0)
            self.sample_sync = pd.concat([self.sample_sync, pd.DataFrame(p_sample_sync)], axis=0)

        self.full.columns = ['idx', 'time'] + ['ts_' + str(i) for i in range(self.num_series)]
        self.sample_async.columns = ['idx', 'time'] + ['ts_' + str(i) for i in range(self.num_series)] + \
                                    ['mask_' + str(i) for i in range(self.num_series)]
        self.sample_sync.columns = ['idx', 'time'] + ['ts_' + str(i) for i in range(self.num_series)] + \
                                   ['mask_' + str(i) for i in range(self.num_series)]

    def sampling_async(self, p):
        p_sample = copy.deepcopy(p)
        if len(p_sample.shape) == 1:
            p_sample = np.expand_dims(p_sample, axis=1)

        mask = np.ones_like(p_sample)
        for c in range(p_sample.shape[1]):
            missing_idx = np.random.choice(self.t, int(self.t * self.args.missing_rate), replace=False)
            p_sample[missing_idx, c] = 0
            mask[missing_idx, c] = 0
        return np.concatenate([p_sample, mask], axis=1)

    def sampling_sync(self, p):
        p_sample = copy.deepcopy(p)
        if len(p_sample.shape) == 1:
            p_sample = np.expand_dims(p_sample, axis=1)

        mask = np.ones_like(p_sample)
        missing_idx = np.random.choice(self.t, int(self.t * self.args.missing_rate), replace=False)
        p_sample[missing_idx, :] = 0
        mask[missing_idx, :] = 0
        return np.concatenate([p_sample, mask], axis=1)

    def plot(self, num_samples):
        plot_idx = np.random.choice(self.N, num_samples)
        plt.figure(figsize=(5 * num_samples, 5), dpi=80)
        for c in range(len(plot_idx)):
            plt.subplot(1, num_samples, c + 1)
            for i in range(self.num_series):
                plt.plot(np.linspace(0, self.T, self.t), self.full[self.full['idx'] == plot_idx[c]]['ts_' + str(i)],
                         label='ts_' + str(i))
            plt.legend()
        plt.show()

    def save(self, data_dirs):
        self.full.to_csv(os.path.join(data_dirs, self.args.type + '_full_data.csv'), index=False)
        self.sample_async.to_csv(os.path.join(data_dirs, 'async_sample_data.csv'), index=False)
        self.sample_sync.to_csv(os.path.join(data_dirs, 'sync_sample_data.csv'), index=False)

class sim_dataset(Dataset):
    def __init__(self, ts):
        assert ts is not None

        # data format: [index, time, ts_1, ts_2..., mask_1, mask_2...]
        # Extract the time series with certain index and reindex the time series
        self.ts = ts.copy()
        # the number of unique index of time series
        self.length = len(np.unique(np.array(self.ts['idx'])))

        map_dict = dict(zip(self.ts["idx"].unique(), np.arange(self.ts["idx"].nunique())))
        self.ts["idx"] = self.ts["idx"].map(map_dict)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_ts = self.ts[self.ts['idx'] == idx]
        return batch_ts


def sim_collate_fn(batch):
    batch_ts = pd.concat(b for b in batch)

    # all sample index in this batch
    sample_idx = pd.unique(batch_ts['idx'])

    # create a obs_idx which indicates the index of samples in this batch
    map_dict = dict(zip(sample_idx, np.arange(len(sample_idx))))
    for idx in sample_idx:
        batch_ts.loc[batch_ts['idx'] == idx, 'batch_idx'] = map_dict[idx]

    # sort the data according by time sequentially
    batch_ts.sort_values(by=['time'], inplace=True)

    # calculating number of events at every time
    obs_times, counts = np.unique(batch_ts.time.values, return_counts=True)
    event_pt = np.concatenate([[0], np.cumsum(counts)])

    # convert data to tensor
    X = torch.FloatTensor(batch_ts.loc[:, [c.startswith("ts") for c in batch_ts.columns]].values)
    M = torch.FloatTensor(batch_ts.loc[:, [c.startswith("mask") for c in batch_ts.columns]].values)
    batch_idx = torch.FloatTensor(batch_ts.loc[:, 'batch_idx'].values)

    res = {}
    res["sample_idx"] = sample_idx
    res["obs_times"] = obs_times
    res["event_pt"] = event_pt
    res["X"] = X
    res["M"] = M
    res["batch_idx"] = batch_idx

    return res

def standard_normal_logprob(z):
    if len(z.shape) == 1:
        z = z.unsqueeze(1)
    d = z.shape[1]
    logZ = -d/2 * math.log(2 * math.pi)
    return logZ - torch.sum(z.pow(2), dim=1, keepdim=True) / 2

def total_derivative(x, t, logp, dx, dlogp, unused_context):
    del logp, dlogp, unused_context

    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]

    try:
        u = torch.full_like(dx, 1/x.numel(), requires_grad=True)
        tmp = torch.autograd.grad( (u*dx).sum(), t, create_graph=True)[0]
        partial_dt = torch.autograd.grad(tmp.sum(), u, create_graph=True)[0]

        total_deriv = directional_dx + partial_dt
    except RuntimeError as e:
        if 'One of the differentiated Tensors' in e.__str__():
            raise RuntimeError('No partial derivative with respect to time. Use mathematically equivalent "directional_derivative" regularizer instead')

    tdv2 = total_deriv.pow(2).view(x.size(0), -1)

    return 0.5*tdv2.mean(dim=-1)

def directional_derivative(x, t, logp, dx, dlogp, unused_context):
    del t, logp, dlogp, unused_context

    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]
    ddx2 = directional_dx.pow(2).view(x.size(0),-1)

    return 0.5*ddx2.mean(dim=-1)

def quadratic_cost(x, t, logp, dx, dlogp, unused_context):
    del x, logp, dlogp, t, unused_context
    dx = dx.view(dx.shape[0], -1)
    return 0.5*dx.pow(2).mean(dim=-1)

def jacobian_frobenius_regularization_fn(x, t, logp, dx, dlogp, context):
    sh = x.shape
    del logp, dlogp, t, dx, x
    sqjac = context.sqjacnorm

    return context.sqjacnorm

def creat_reg_fns(args):
    REGULARIZATION_FNS = {
        "kinetic_energy": quadratic_cost,
        "jacobian_norm2": jacobian_frobenius_regularization_fn,
        "directional_penalty": directional_derivative
    }

    reg_fns = []
    reg_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if getattr(args, arg_key) is not None:
            reg_fns.append(reg_fn)
            reg_coeffs.append(eval("args." + arg_key))

    reg_fns = tuple(reg_fns)
    reg_coeffs = tuple(reg_coeffs)
    return reg_fns, reg_coeffs

def get_transforms(model):

    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn


def viz_flow(t, x):
    """Produces visualization for the model density and samples from the model."""
    if len(x.shape) == 3:
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    df_corr = pd.DataFrame(x.detach().cpu().numpy()).corr()

    sns.heatmap(df_corr, center=0, annot=True, cmap='YlGnBu')
    plt.title('Predicted correlation matrix at time '+str(t))


def get_transforms_test(model):

    def sample_fn(z, h, logpz=None):
        if logpz is not None:
            return model(z, h, logpz, reverse=True)
        else:
            return model(z, h, reverse=True)

    def density_fn(x, h, logpx=None):
        if logpx is not None:
            return model(x, h, logpx, reverse=False)
        else:
            return model(x, h, reverse=False)

    return sample_fn, density_fn

def viz_flow_test(num_sampling_samples, t, base_sample, sample_fn, mgn_h, ax, plot_idx, device='cpu'):
    """Produces visualization for the model density and samples from the model."""
    z = base_sample.sample(sample_shape=torch.Size([num_sampling_samples])).type(torch.float32).to(device)
    x = sample_fn(z.unsqueeze(1), mgn_h).detach().cpu().numpy().squeeze(1)
    f = sns.distplot(np.array(x), bins=50, hist=True, rug=True, ax=ax[1, plot_idx])
    f.set(title=f'Flow Samples density at t={t}')


def viz_GRUODE(t, x):
    """Produces visualization for the model density and samples from the model."""
    df_corr = pd.DataFrame(x).corr()
    sns.heatmap(df_corr, center=0, annot=True, cmap='YlGnBu')
    plt.title(f'Predicted correlation matrix at time point '+str(t))

def plot_density(full_data, t_list=[0.5]):
    sample_list = []
    for t in t_list:
        sample = full_data[full_data['time'] == t]['ts_1']
        sample_list.append(sample)

    sns.set_theme()
    fig, ax = plt.subplots(1, len(t_list), figsize=(15, 10))

    for l, t in enumerate(t_list):
        f = sns.distplot(np.array(sample_list[l]), bins=50, hist=True, rug=True, ax=ax[l])
        f.set(title=f'Sample distribution for t={t}')

    plt.show()

def compute_CRPS(x, X_obs, M_obs):
    crps_list = []

    for i in range(x.shape[1]):
        for d in range(x.shape[-1]):
            if M_obs[i, d] == 1:
                crps_list.append(ps.crps_ensemble(X_obs[i, d], x[:, i, d]))
    return crps_list

def compute_CRPS_sum(x, X_obs, M_obs):
    crps_sum_list = []
    x = x * M_obs
    x = np.sum(x, axis=2)
    X_obs = np.sum(X_obs, axis=1)
    for i in range(x.shape[1]):
        crps_sum_list.append(ps.crps_ensemble(X_obs[i], x[:, i]))
    return crps_sum_list

def compute_ND(x, X_obs, M_obs):
    ND_err, ND_all = [], []

    for i in range(x.shape[1]):
        for d in range(x.shape[-1]):
            if M_obs[i, d] == 1:
                ND_err.append(np.abs(X_obs[i, d]-np.median(x[:, i, d])))
                ND_all.append(np.abs(X_obs[i, d]))
    return np.sum(ND_err), np.sum(ND_all)

