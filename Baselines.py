import numpy as np
import torch
import utils
import torch.nn as nn
from torchdiffeq import odeint
import math
from torch.distributions.normal import Normal
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain
from torch.nn import Parameter


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

'''
This part of code are mainly implemented according GRU-ODE-Bayes
https://arxiv.org/abs/1905.12374
'''

class GRUODECell(torch.nn.Module):

    def __init__(self, input_size, n_dim, bias=True):
        super().__init__()
        hidden_size = input_size * n_dim
        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.apply(init_weights)

    def forward(self, t, h):
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))
        dh = (1 - z) * (u - h)
        return dh


class GRUObsCell(torch.nn.Module):

    def __init__(self, input_size, n_dim, bias=True):
        super().__init__()
        self.input_size = input_size
        self.n_dim = n_dim
        self.hidden_size = input_size * n_dim
        self.gru_d = nn.GRUCell(self.input_size, self.hidden_size, bias=bias)

    def forward(self, h, X_obs, M_obs, i_obs):
        # only compute losses on observations
        temp = h.clone()
        temp[i_obs] = self.gru_d(X_obs, h[i_obs])
        h = temp

        return h


class GRU_ODE(nn.Module):
    def __init__(self, args, device):
        super(GRU_ODE, self).__init__()

        # params of GRU_ODE Networks
        self.hidden_size = args.input_dim * args.memory_dim
        self.input_size = args.input_dim
        self.solver = args.solver
        self.dropout = args.dropout
        self.atol = args.atol
        self.rtol = args.rtol
        self.n_dim = args.memory_dim

        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size, 2 * self.input_size, bias=True)
        )

        # GRU-ODE
        self.gru_c = GRUODECell(self.input_size, self.n_dim, bias=True)
        # GRU-BAYES
        self.gru_obs = GRUObsCell(self.input_size, self.n_dim, bias=True)

        assert self.solver in ["euler", "midpoint", "rk4", "explicit_adams", "implicit_adams",
                               "dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun"]

        self.apply(init_weights)

    def ode_step(self, h, delta_t):
        solution = odeint(self.gru_c, h, torch.tensor([0, delta_t]).to(h.device), method=self.solver, atol=self.atol, rtol=self.rtol)
        h = solution[1, :, :]
        p = self.p_model(h)
        return h, p

    def base_dist(self, mean, var):
        return Normal(mean, var)

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, device, dt=0.01, viz=False, val=False, t_corr=[0.5]):

        # create the hidden state for each sampled time series
        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(device)
        # create the mapping function from hidden state to real data
        p = self.p_model(h)

        current_time = 0.0
        loss = torch.as_tensor(0.0)
        total_M_obs = 0

        if val:
            num_sampling_samples = 100
            crps_list, crps_sum_list = [], []
            ND_err, ND_all = 0, 0
            if viz:
                sns.set_theme()
                sns.set_theme()
                plt.figure(figsize=[5 * len(t_corr), 5])
                plot_idx = 1

        for i, obs_time in enumerate(obs_times):
            # do not reach the observation, using ODE to update hidden state
            while current_time < obs_time:
                h, p = self.ode_step(h, dt)
                current_time = current_time + dt

            # Reached an observation, using GRU cell to update hidden state
            start = event_pt[i]
            end = event_pt[i + 1]

            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = batch_idx[start:end].type(torch.LongTensor)
            p = self.p_model(h)
            p_obs = p[i_obs]

            # compute loss
            mean, logvar = torch.chunk(p_obs, 2, dim=1)
            sigma = torch.exp(0.5 * logvar)
            error = (X_obs - mean) / sigma
            log_lik_c = np.log(np.sqrt(2 * np.pi))
            losses = 0.5 * ((torch.pow(error, 2) + logvar + 2 * log_lik_c) * M_obs)

            if val:
                x = Normal(mean, sigma).sample(sample_shape=torch.Size([num_sampling_samples])).type(torch.float32).to(device)
                crps = utils.compute_CRPS(x.cpu().detach().numpy(),
                                          X_obs.cpu().detach().numpy(),
                                          M_obs.cpu().detach().numpy())
                crps_sum = utils.compute_CRPS_sum(x.cpu().detach().numpy(),
                                                  X_obs.cpu().detach().numpy(),
                                                  M_obs.cpu().detach().numpy())
                ND_err_t, ND_all_t = utils.compute_ND(x.cpu().detach().numpy(),
                                                      X_obs.cpu().detach().numpy(),
                                                      M_obs.cpu().detach().numpy())
                crps_list.append(crps)
                crps_sum_list.append(crps_sum)
                ND_err += ND_err_t
                ND_all += ND_all_t

                if viz:
                    if np.round(current_time, 2) in t_corr:
                        plt.subplot(1, len(t_corr), plot_idx)
                        utils.viz_GRUODE(np.round(current_time, 2), x.reshape(-1, x.shape[-1]).detach().cpu().numpy())
                        plot_idx += 1

            # Using GRUObservationCell to update h.
            h = self.gru_obs(h, X_obs, M_obs, i_obs)

            loss = loss + losses.sum()
            total_M_obs = total_M_obs + M_obs.sum()


        if val:
            return loss / total_M_obs, np.mean(list(chain(*crps_list))), np.mean(list(chain(*crps_sum_list))), ND_err/ND_all
        else:
            return loss / total_M_obs, torch.tensor(0.0)



'''
This part of code are mainly implemented according ODE-LSTM
https://arxiv.org/pdf/2006.04418.pdf
'''


class ODENetTS(nn.Module):
    def __init__(self, input_size, n_dim):
        super(ODENetTS, self).__init__()
        self.input_size = input_size
        self.n_dim = n_dim
        self.hidden_size = input_size * n_dim
        # hidden state evolement by an ODE
        self.ODEFunc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.reset_parameters()

    def forward(self, t, h):
        h = self.ODEFunc(h)
        return h

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.0)
            elif 'w' in name:
                nn.init.xavier_uniform_(param.data)



class ODELSTMCell(nn.Module):
    def __init__(self, input_size, n_dim, bias, atol, rtol):
        super(ODELSTMCell, self).__init__()
        self.input_size = input_size
        self.n_dim = n_dim
        self.hidden_size = input_size * n_dim

        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size, bias=bias)

        self.atol = atol
        self.rtol = rtol

        # 1 hidden layer NODE
        self.ODEFunc = ODENetTS(self.input_size, self.n_dim)
        self.apply(init_weights)

    def forward(self, X, M, h, c, delta_t, solver, update):
        if update:
            h, c = self.lstm(X, (h, c))
            return h, c
        else:
            h = odeint(self.ODEFunc, h, torch.tensor([0, delta_t]).to(h.device), method=solver, atol=self.atol, rtol=self.rtol)[1]
            return h


class ODELSTM(nn.Module):
    def __init__(self, args, device):
        super(ODELSTM, self).__init__()

        self.n_dim = args.memory_dim
        self.hidden_size = args.input_dim * args.memory_dim
        self.cell_size = args.input_dim * args.memory_dim
        self.input_size = args.input_dim
        self.solver = args.solver
        self.dropout = args.dropout

        # ODE-LSTM Cell
        self.odelstm = ODELSTMCell(self.input_size, self.n_dim, bias=True, atol=args.atol, rtol=args.rtol)

        # mapping function from hidden state to real data
        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size, 2 * self.input_size, bias=True)
        )

        self.apply(init_weights)

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, device, dt=0.01, viz=False, val=False,
                t_corr=[0.5]):

        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(device)
        c = torch.zeros([sample_idx.shape[0], self.cell_size]).to(device)
        # create the mapping function from hidden state to real data

        current_time = 0
        loss = torch.as_tensor(0.0)
        total_M_obs = 0

        if val:
            num_sampling_samples = 100
            crps_list, crps_sum_list = [], []
            ND_err, ND_all = 0, 0
            if viz:
                sns.set_theme()
                sns.set_theme()
                plt.figure(figsize=[5 * len(t_corr), 5])
                plot_idx = 1

        for i, obs_time in enumerate(obs_times):
            # do not reach the observation, using ODE to update hidden state
            while current_time < obs_time:
                h = self.odelstm(None, None, h, c, dt, self.solver, update=False)

                current_time = current_time + dt

            start = event_pt[i]
            end = event_pt[i + 1]

            X_obs = X[start:end, :]
            M_obs = M[start:end, :]
            i_obs = batch_idx[start:end].type(torch.LongTensor)
            p = self.p_model(h)
            p_obs = p[i_obs]

            # compute loss
            mean, logvar = torch.chunk(p_obs, 2, dim=1)
            sigma = torch.exp(0.5 * logvar)
            error = (X_obs - mean) / sigma
            log_lik_c = np.log(np.sqrt(2 * np.pi))
            losses = 0.5 * ((torch.pow(error, 2) + logvar + 2 * log_lik_c) * M_obs)

            if val:
                x = Normal(mean, sigma).sample(sample_shape=torch.Size([num_sampling_samples])).type(torch.float32).to(
                    device)
                crps = utils.compute_CRPS(x.cpu().detach().numpy(),
                                          X_obs.cpu().detach().numpy(),
                                          M_obs.cpu().detach().numpy())
                crps_sum = utils.compute_CRPS_sum(x.cpu().detach().numpy(),
                                                  X_obs.cpu().detach().numpy(),
                                                  M_obs.cpu().detach().numpy())
                ND_err_t, ND_all_t = utils.compute_ND(x.cpu().detach().numpy(),
                                                      X_obs.cpu().detach().numpy(),
                                                      M_obs.cpu().detach().numpy())
                crps_list.append(crps)
                crps_sum_list.append(crps_sum)
                ND_err += ND_err_t
                ND_all += ND_all_t

                if viz:
                    if np.round(current_time, 2) in t_corr:
                        plt.subplot(1, len(t_corr), plot_idx)
                        utils.viz_GRUODE(np.round(current_time, 2), x.reshape(-1, x.shape[-1]).detach().cpu().numpy())
                        plot_idx += 1

            # update the hidden state and cell state
            temp_c, temp_h = c.clone(), h.clone()
            # if there exists observations, using LSTM updated
            temp_h[i_obs], temp_c[i_obs] = self.odelstm(X_obs, M_obs, h[i_obs], c[i_obs], dt, self.solver, update=True)

            c, h = temp_c, temp_h

            loss = loss + losses.sum()
            total_M_obs = total_M_obs + M_obs.sum()

        if val:
            return loss / total_M_obs, np.mean(list(chain(*crps_list))), np.mean(list(chain(*crps_sum_list))), ND_err / ND_all
        else:
            return loss / total_M_obs, torch.tensor(0.0)


'''
This part of code are mainly implemented according ODE-LSTM
https://arxiv.org/pdf/1907.03907.pdf
'''

class ODERNNCell(nn.Module):
    def __init__(self, input_size, n_dim, bias, atol, rtol):
        super(ODERNNCell, self).__init__()
        self.input_size = input_size
        self.n_dim = n_dim
        self.hidden_size = input_size * n_dim
        self.atol = atol
        self.rtol = rtol

        self.rnn = nn.GRUCell(self.input_size, self.hidden_size, bias=bias)
        # 1 hidden layer NODE
        self.ODEFunc = ODENetTS(self.input_size, self.n_dim)
        self.apply(init_weights)

    def forward(self, X, M, h, delta_t, solver, update):
        if update:
            h = self.rnn(X, h)
            return h
        else:
            h = odeint(self.ODEFunc, h, torch.tensor([0, delta_t]).to(h.device), method=solver, atol=self.atol, rtol=self.rtol)[1]
            return h


class ODERNN(nn.Module):
    def __init__(self, args, device):
        super(ODERNN, self).__init__()

        self.n_dim = args.memory_dim
        self.hidden_size = args.input_dim * args.memory_dim
        self.input_size = args.input_dim
        self.solver = args.solver
        self.dropout = args.dropout

        # ODE-RNN Cell
        self.odernn = ODERNNCell(self.input_size, self.n_dim, bias=True, atol=args.atol, rtol=args.rtol)

        # mapping function from hidden state to real data
        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size, 2 * self.input_size, bias=True)
        )

        self.apply(init_weights)

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, device, dt=0.01, viz=False, val=False,
                t_corr=[0.5]):

        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(device)
        # create the mapping function from hidden state to real data

        current_time = 0
        loss = torch.as_tensor(0.0)
        total_M_obs = 0

        if val:
            num_sampling_samples = 100
            crps_list, crps_sum_list = [], []
            ND_err, ND_all = 0, 0
            if viz:
                sns.set_theme()
                sns.set_theme()
                plt.figure(figsize=[5 * len(t_corr), 5])
                plot_idx = 1

        for i, obs_time in enumerate(obs_times):
            # do not reach the observation, using ODE to update hidden state
            while current_time < obs_time:
                h = self.odernn(None, None, h, dt, self.solver, update=False)

                current_time = current_time + dt

            start = event_pt[i]
            end = event_pt[i + 1]

            X_obs = X[start:end, :]
            M_obs = M[start:end, :]
            i_obs = batch_idx[start:end].type(torch.LongTensor)
            p = self.p_model(h)
            p_obs = p[i_obs]

            # compute loss
            mean, logvar = torch.chunk(p_obs, 2, dim=1)
            sigma = torch.exp(0.5 * logvar)
            error = (X_obs - mean) / sigma
            log_lik_c = np.log(np.sqrt(2 * np.pi))
            losses = 0.5 * ((torch.pow(error, 2) + logvar + 2 * log_lik_c) * M_obs)

            if val:
                x = Normal(mean, sigma).sample(sample_shape=torch.Size([num_sampling_samples])).type(torch.float32).to(
                    device)
                crps = utils.compute_CRPS(x.cpu().detach().numpy(),
                                          X_obs.cpu().detach().numpy(),
                                          M_obs.cpu().detach().numpy())
                crps_sum = utils.compute_CRPS_sum(x.cpu().detach().numpy(),
                                                  X_obs.cpu().detach().numpy(),
                                                  M_obs.cpu().detach().numpy())
                ND_err_t, ND_all_t = utils.compute_ND(x.cpu().detach().numpy(),
                                                      X_obs.cpu().detach().numpy(),
                                                      M_obs.cpu().detach().numpy())
                crps_list.append(crps)
                crps_sum_list.append(crps_sum)
                ND_err += ND_err_t
                ND_all += ND_all_t

                if viz:
                    if np.round(current_time, 2) in t_corr:
                        plt.subplot(1, len(t_corr), plot_idx)
                        utils.viz_GRUODE(np.round(current_time, 2), x.reshape(-1, x.shape[-1]).detach().cpu().numpy())
                        plot_idx += 1

            # update the hidden state and cell state
            temp_h = h.clone()
            # if there exists observations, using LSTM updated
            temp_h[i_obs] = self.odernn(X_obs, M_obs, h[i_obs], dt, self.solver, update=True)
            h = temp_h

            loss = loss + losses.sum()
            total_M_obs = total_M_obs + M_obs.sum()

        if val:
            return loss / total_M_obs, np.mean(list(chain(*crps_list))), np.mean(list(chain(*crps_sum_list))), ND_err / ND_all
        else:
            return loss / total_M_obs, torch.tensor(0.0)

'''
This part of code are mainly implemented according GRU-D
https://arxiv.org/abs/1606.01865
'''

class GRU_D_cell(nn.Module):
    def __init__(self, input_size, n_dim):
        super(GRU_D_cell, self).__init__()
        self.input_size = input_size
        self.n_dim = n_dim
        self.hidden_size = input_size * n_dim

        self.W_r = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.V_r = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.U_r = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))

        self.W_z = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.V_z = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.U_z = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))

        self.W_h = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.V_h = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.U_h = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))

        self.b_r = nn.Parameter(torch.randn(self.hidden_size))
        self.b_z = nn.Parameter(torch.randn(self.hidden_size))
        self.b_h = nn.Parameter(torch.randn(self.hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.0)
            elif 'W' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'U' in name:
                nn.init.orthogonal_(param.data)
            elif 'V' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, h, X_hat, M_obs, gamma_h):
        h = gamma_h * h
        r = torch.sigmoid(torch.mm(X_hat, self.W_r) + torch.mm(h, self.U_r) + torch.mm(M_obs, self.V_r) + self.b_r)
        z = torch.sigmoid(torch.mm(X_hat, self.W_z) + torch.mm(h, self.U_z) + torch.mm(M_obs, self.V_z) + self.b_z)
        h_tilde = torch.tanh(
            torch.mm(X_hat, self.W_h) + torch.mm(r * h, self.U_h) + torch.mm(M_obs, self.V_h) + self.b_h)
        h = (1 - z) * h + z * h_tilde

        return h


class GRU_D(nn.Module):
    def __init__(self, args, device):
        super(GRU_D, self).__init__()

        self.input_size = args.input_dim
        self.n_dim = args.memory_dim
        self.hidden_size = args.input_dim * args.memory_dim
        self.dropout = args.dropout

        # mapping function from hidden state to real data
        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size, 2 * self.input_size, bias=True)
        )

        self.gru_d = GRU_D_cell(self.input_size, self.n_dim)
        # decay parameters
        self.lin_gamma_x = nn.Linear(self.input_size, self.input_size, bias=False)
        self.lin_gamma_h = nn.Linear(self.input_size, self.hidden_size, bias=False)

        self.apply(init_weights)

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, device, dt=0.01, viz=False, val=False,
                t_corr=[0.5]):

        # create the hidden state for each sampled time series
        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(device)
        p = self.p_model(h)
        # create and store the last observation time and value for each sampled time series
        last_t = torch.zeros([sample_idx.shape[0], self.input_size]).to(device)
        last_x = torch.zeros([sample_idx.shape[0], self.input_size]).to(device)

        current_time = 0
        loss = torch.as_tensor(0.0)
        total_M_obs = 0

        if val:
            num_sampling_samples = 100
            crps_list, crps_sum_list = [], []
            ND_err, ND_all = 0, 0
            if viz:
                sns.set_theme()
                sns.set_theme()
                plt.figure(figsize=[5 * len(t_corr), 5])
                plot_idx = 1

        for i, obs_time in enumerate(obs_times):
            current_time = obs_time
            start = event_pt[i]
            end = event_pt[i + 1]

            X_obs = X[start:end, :]
            M_obs = M[start:end, :]
            i_obs = batch_idx[start:end].type(torch.LongTensor)

            p = self.p_model(h)
            p_obs = p[i_obs]

            # compute loss
            mean, logvar = torch.chunk(p_obs, 2, dim=1)
            sigma = torch.exp(0.5 * logvar)
            error = (X_obs - mean) / sigma
            log_lik_c = np.log(np.sqrt(2 * np.pi))
            losses = 0.5 * ((torch.pow(error, 2) + logvar + 2 * log_lik_c) * M_obs)

            if val:
                x = Normal(mean, sigma).sample(sample_shape=torch.Size([num_sampling_samples])).type(torch.float32).to(
                    device)

                crps = utils.compute_CRPS(x.cpu().detach().numpy(),
                                          X_obs.cpu().detach().numpy(),
                                          M_obs.cpu().detach().numpy())
                crps_sum = utils.compute_CRPS_sum(x.cpu().detach().numpy(),
                                                  X_obs.cpu().detach().numpy(),
                                                  M_obs.cpu().detach().numpy())
                ND_err_t, ND_all_t = utils.compute_ND(x.cpu().detach().numpy(),
                                                      X_obs.cpu().detach().numpy(),
                                                      M_obs.cpu().detach().numpy())
                crps_list.append(crps)
                crps_sum_list.append(crps_sum)
                ND_err += ND_err_t
                ND_all += ND_all_t

                if viz:
                    if np.round(current_time, 2) in t_corr:
                        plt.subplot(1, len(t_corr), plot_idx)
                        utils.viz_GRUODE(np.round(current_time, 2), x.reshape(-1, x.shape[-1]).detach().cpu().numpy())
                        plot_idx += 1
            # update saved X at the last time point
            last_x[i_obs, :] = last_x[i_obs, :] * (1 - M_obs) + X_obs * M_obs
            # compute the mean of each variables
            mean_x = torch.sum(X_obs, dim=0, keepdim=True) / torch.sum(M_obs+1e-6, dim=0, keepdim=True)
            # get the time interval
            interval = obs_time - last_t
            # update the time point of last observation
            last_t[i_obs, :] = last_t[i_obs, :] * (1 - M_obs) + obs_time * M_obs

            gamma_x = torch.exp(-torch.maximum(torch.tensor(0.0).to(device), self.lin_gamma_x(interval)))
            gamma_h = torch.exp(-torch.maximum(torch.tensor(0.0).to(device), self.lin_gamma_h(interval)))

            X_hat = M_obs * X_obs + (1 - M_obs) * gamma_x[i_obs] * last_x[i_obs] + \
                    (1 - M_obs) * (1 - gamma_x[i_obs]) * mean_x

            temp = h.clone()
            temp[i_obs] = self.gru_d(h[i_obs], X_hat, M_obs, gamma_h[i_obs])
            h = temp

            loss = loss + losses.sum()
            total_M_obs = total_M_obs + M_obs.sum()

        if val:
            return loss / total_M_obs, np.mean(list(chain(*crps_list))), np.mean(list(chain(*crps_sum_list))), ND_err / ND_all
        else:
            return loss / total_M_obs, torch.tensor(0.0)
