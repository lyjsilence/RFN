import numpy as np
import utils
import torch
import torch.nn as nn
from torch.nn import Parameter
from torchdiffeq import odeint_adjoint as odeint
from Baselines import GRU_ODE, ODERNN, ODELSTM, GRU_D
from torch.distributions.normal import Normal
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from itertools import chain


'''When the marginal learning block is GRU-ODE-Bayes'''
class mgn_GRUODEObsCell(nn.Module):
    def __init__(self, input_size, n_dim, bias=True):
        super().__init__()
        self.n_dim = n_dim
        self.GRUCell = nn.GRUCell(input_size, input_size*n_dim, bias=bias)

    def forward(self, mgn_h, X_obs, M_obs, i_obs):
        temp_h = mgn_h.clone()
        temp_h[i_obs] = self.GRUCell(X_obs, mgn_h[i_obs])
        mgn_h = temp_h

        return mgn_h


class mgn_GRUODE(nn.Module):
    def __init__(self, args, device):
        super(mgn_GRUODE, self).__init__()
        self.solver = args.solver
        self.device = device
        self.atol = args.atol
        self.rtol = args.rtol

        # mgn_ode updates h by ODE function without data
        # gru_update updates h and h_tilde by GRU cell with observed data
        from Baselines import GRUODECell
        self.mgn_ode = GRUODECell(args.input_dim, args.memory_dim)
        self.mgn_update = mgn_GRUODEObsCell(args.input_dim, args.memory_dim)

    def forward(self, current_time, mgn_h, delta_t, X_obs=None, M_obs=None, i_obs=None, update=False):
        if update:
            assert X_obs is not None
            assert M_obs is not None
            assert i_obs is not None

            mgn_h = self.mgn_update(mgn_h, X_obs, M_obs, i_obs)
        else:
            t_list = torch.cat([torch.as_tensor(current_time).unsqueeze(0),
                                torch.as_tensor(current_time + delta_t).unsqueeze(0)]).to(self.device)
            mgn_h = odeint(self.mgn_ode, mgn_h, t_list, method=self.solver, atol=self.atol, rtol=self.rtol)[1]
        return mgn_h


'''When the marginal learning block is ODELSTM'''

class mgn_ODELSTM(nn.Module):
    def __init__(self, args, device):
        super(mgn_ODELSTM, self).__init__()
        self.input_size = args.input_dim
        self.n_dim = args.memory_dim
        self.hidden_size = self.input_size * self.n_dim
        self.solver = args.solver
        self.device = device
        self.atol = args.atol
        self.rtol = args.rtol

        from Baselines import ODENetTS
        # hidden state update
        self.lstm = nn.LSTMCell(self.input_size, self.input_size * self.n_dim, bias=True)
        # hidden state evolvement
        self.ODElstm = ODENetTS(self.input_size, self.n_dim)

    def forward(self, current_time, mgn_hc, delta_t, X_obs=None, M_obs=None, i_obs=None, update=False):
        mgn_h, mgn_c = mgn_hc[0], mgn_hc[1]
        if update:
            temp_h, temp_c = mgn_h.clone(), mgn_c.clone()
            temp_h[i_obs], temp_c[i_obs] = self.lstm(X_obs, (mgn_h[i_obs], mgn_c[i_obs]))
            mgn_h, mgn_c = temp_h, temp_c
            return mgn_h, mgn_c
        else:
            t_list = torch.cat([torch.as_tensor(current_time).unsqueeze(0),
                                torch.as_tensor(current_time + delta_t).unsqueeze(0)]).to(self.device)
            mgn_h = odeint(self.ODElstm, mgn_h, t_list, method=self.solver, atol=self.atol, rtol=self.rtol)[1]

        return mgn_h, mgn_c


'''When the marginal learning block is ODERNN'''

class mgn_ODERNN(nn.Module):
    def __init__(self, args, device):
        super(mgn_ODERNN, self).__init__()
        self.input_size = args.input_dim
        self.n_dim = args.memory_dim
        self.hidden_size = self.input_size * self.n_dim
        self.solver = args.solver
        self.device = device
        self.atol = args.atol
        self.rtol = args.rtol

        from Baselines import ODENetTS
        # hidden state update
        self.rnn = nn.GRUCell(self.input_size, self.input_size * self.n_dim, bias=True)
        # hidden state evolvement
        self.odernn = ODENetTS(self.input_size, self.n_dim)

    def forward(self, current_time, mgn_h, delta_t, X_obs=None, M_obs=None, i_obs=None, update=False):
        if update:
            temp_h = mgn_h.clone()
            temp_h[i_obs] = self.rnn(X_obs, mgn_h[i_obs])
            mgn_h = temp_h
            return mgn_h
        else:
            t_list = torch.cat([torch.as_tensor(current_time).unsqueeze(0),
                                torch.as_tensor(current_time + delta_t).unsqueeze(0)]).to(self.device)
            mgn_h = odeint(self.odernn, mgn_h, t_list, method=self.solver, atol=self.atol, rtol=self.rtol)[1]

        return mgn_h


'''When the marginal learning block is GRU-D'''
class mgn_GRU_D(nn.Module):
    def __init__(self, args, device):
        super(mgn_GRU_D, self).__init__()
        self.input_size = args.input_dim
        self.n_dim = args.memory_dim
        self.hidden_size = self.input_size * self.n_dim
        self.solver = args.solver
        self.device = device

        from Baselines import GRU_D_cell
        self.GRU_D = GRU_D_cell(self.input_size, self.n_dim)
        # decay parameters
        self.lin_gamma_x = nn.Linear(self.input_size, self.input_size, bias=False)
        self.lin_gamma_h = nn.Linear(self.input_size, self.hidden_size, bias=False)

    def forward(self, current_time, mgn_h, X_obs=None, M_obs=None, i_obs=None, last_x=None, last_t=None):
        # update saved X at the last time point
        last_x[i_obs, :] = last_x[i_obs, :] * (1 - M_obs) + X_obs * M_obs
        # compute the mean of each variables
        mean_x = torch.sum(X_obs, dim=0, keepdim=True) / torch.sum(M_obs+1e-6, dim=0, keepdim=True)
        # get the time interval
        interval = current_time - last_t
        # update the time point of last observation
        last_t[i_obs, :] = last_t[i_obs, :] * (1 - M_obs) + current_time * M_obs

        gamma_x = torch.exp(-torch.maximum(torch.tensor(0.0).to(self.device), self.lin_gamma_x(interval)))
        gamma_h = torch.exp(-torch.maximum(torch.tensor(0.0).to(self.device), self.lin_gamma_h(interval)))

        X_hat = M_obs * X_obs + (1 - M_obs) * gamma_x[i_obs] * last_x[i_obs] + (1 - M_obs) * (1 - gamma_x[i_obs]) * mean_x

        temp = mgn_h.clone()
        temp[i_obs] = self.GRU_D(mgn_h[i_obs], X_hat, M_obs, gamma_h[i_obs])
        mgn_h = temp

        return mgn_h, last_x, last_t


'''Dependence learning block'''
class Gated_Linear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Gated_Linear, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.bias = nn.Linear(1, dim_out, bias=False)
        self.gate = nn.Linear(1, dim_out)

    def forward(self, s, x):
        '''The vector field of flow model is controled by the physical time t and flow time s'''
        gate = torch.sigmoid(self.gate(s))
        return self.linear(x) * gate + self.bias(s)


class ODENet(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, hidden_layer=2):
        super(ODENet, self).__init__()
        # input layer
        linear_layers = [Gated_Linear(input_dim+cond_dim, hidden_dim)]
        act_layers = [nn.Softplus()]
        # hidden layer
        for i in range(hidden_layer):
            linear_layers.append(Gated_Linear(hidden_dim, hidden_dim))
            act_layers.append(nn.Softplus())
        # output layer
        linear_layers.append(Gated_Linear(hidden_dim, input_dim))

        self.linear_layers = nn.ModuleList(linear_layers)
        self.act_layers = nn.ModuleList(act_layers)

    def forward(self, s, x, m, cond=None):
        assert cond is not None
        dx = torch.cat([x, cond], dim=1)

        for l, layer in enumerate(self.linear_layers):
            dx = layer(s, dx)
            if l < len(self.linear_layers) - 1:
                dx = self.act_layers[l](dx)
        return dx * m


class ODEFunc(nn.Module):
    def __init__(self, ODENet, input_dim, rademacher=False, div_samples=1):
        super(ODEFunc, self).__init__()

        self.ODENet = ODENet
        self.rademacher = rademacher
        self.div_samples = div_samples
        self.divergence_fn = self.divergence_approx
        self.input_dim = input_dim

    def divergence_approx(self, f, z, m, e=None):
        samples = []
        sqnorms = []
        for e_ in e:
            e_dzdx = torch.autograd.grad(f, z, e_, create_graph=True)[0]  # *m
            n = e_dzdx.view(z.size(0), -1).pow(2).mean(dim=1, keepdim=True)
            sqnorms.append(n)
            e_dzdx_e = e_dzdx * e_
            samples.append(e_dzdx_e.view(z.shape[0], -1).sum(dim=1, keepdim=True))

        S = torch.cat(samples, dim=1)
        approx_tr_dzdx = S.mean(dim=1)

        N = torch.cat(sqnorms, dim=1).mean(dim=1)

        return approx_tr_dzdx, N

    def sample_rademacher(self, y):
        return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1

    def sample_gaussian(self, y):
        return torch.randn_like(y)

    def before_odeint(self, e=None):
        self._e = e
        self._sqjacnorm = None

    def forward(self, s, states):
        assert len(states) >= 2
        z = states[0]
        cond = states[2]

        if len(z.shape) == 1:
            z = z.unsqueeze(1)

        # convert to tensor
        s = s.to(z)

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = [self.sample_rademacher(z) for k in range(self.div_samples)]
            else:
                self._e = [self.sample_gaussian(z) for k in range(self.div_samples)]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            s = torch.ones(z.size(0), 1).to(z) * s.clone().detach().requires_grad_(True).type_as(z)
            m = states[3]
            for s_ in states[4:]:
                s_.requires_grad_(True)

            # compute dz by ODEnet
            dz = self.ODENet(s, z, m, cond=cond, *states[4:])

            # Compute tr(df/dz)
            if not self.training and dz.view(dz.shape[0], -1).shape[1] == 2:
                divergence = self.divergence_bf(dz, z).view(z.shape[0], 1)
            else:
                divergence, sqjacnorm = self.divergence_fn(dz, z, m, e=self._e)
                divergence = divergence.view(z.shape[0], 1)
            self.sqjacnorm = sqjacnorm

        return tuple([dz, -divergence, torch.zeros_like(cond).requires_grad_(True), torch.zeros_like(m).requires_grad_(True)]
                     + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[4:]])


class RegularizedODEfunc(nn.Module):
    def __init__(self, ODEFunc, reg_fns, input_dim):
        super(RegularizedODEfunc, self).__init__()
        self.ODEFunc = ODEFunc
        self.reg_fns = reg_fns
        self.input_dim = input_dim

    def before_odeint(self, *args, **kwargs):
        self.ODEFunc.before_odeint(*args, **kwargs)

    def forward(self, s, state):
        with torch.enable_grad():
            x, logp, cond, m = state[:4]
            x.requires_grad_(True)
            s.requires_grad_(True)
            logp.requires_grad_(True)
            dstate = self.ODEFunc(s, (x, logp, cond, m))
            if len(state) > 4:
                dx, dlogp, cond, m = dstate[:4]
                reg_states = tuple(reg_fn(x, s, logp, dx, dlogp, self.ODEFunc) for reg_fn in self.reg_fns)
                return dstate + reg_states
            else:
                return dstate


class CNF(nn.Module):
    def __init__(self, ODEFunc, input_dim, T=1.0, reg_fns=None, train_T=False, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if reg_fns is not None:
            ODEFunc = RegularizedODEfunc(ODEFunc, reg_fns, input_dim)
            nreg = len(reg_fns)

        self.input_dim = input_dim
        self.nreg = nreg
        self.ODEFunc = ODEFunc
        self.regularization_states = None
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def _flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def forward(self, z_, logpz=None, reg_states=tuple(), integration_times=None, reverse=False):

        z, m, cond = z_[:, :self.input_dim], z_[:, self.input_dim:2*self.input_dim], z_[:, 2*self.input_dim:]

        if not len(reg_states) == self.nreg and self.training:
            reg_states = tuple(torch.zeros(z.size(0)).to(z) for i in range(self.nreg))

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        if reverse:
            integration_times = self._flip(integration_times, 0)

        # Refresh
        self.ODEFunc.before_odeint()

        if self.training:
            state_t = odeint(self.ODEFunc, (z, _logpz, cond, m)+reg_states, integration_times.to(z),
                             atol=self.atol, rtol=self.rtol, method=self.solver, options=self.solver_options)
        else:
            state_t = odeint(self.ODEFunc, (z, _logpz, cond, m), integration_times.to(z),
                             atol=self.test_atol, rtol=self.test_rtol, method=self.test_solver)

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t, cond, m = state_t[:4]
        reg_states = state_t[4:]

        if logpz is not None:
            return torch.cat([z_t, m, cond], dim=1), logpz_t, reg_states
        else:
            return torch.cat([z_t, m, cond], dim=1)


class joint_RFN(nn.Module):
    def __init__(self, args):
        super(joint_RFN, self).__init__()

        reg_fns, reg_coeffs = utils.creat_reg_fns(args)

        chain = [self.build_cnf(args, reg_fns) for _ in range(args.num_blocks)]

        self.reg_coeffs = reg_coeffs

        self.chain = nn.ModuleList(chain)

    def build_cnf(self, args, reg_fns):
        f = ODENet(input_dim=args.input_dim,  cond_dim=args.input_dim*args.memory_dim, hidden_dim=args.hidden_dim)
        f_aug = ODEFunc(ODENet=f, input_dim=args.input_dim, rademacher=args.rademacher)
        cnf = CNF(ODEFunc=f_aug, input_dim=args.input_dim, T=args.flow_time, reg_fns=reg_fns, train_T=args.train_T, solver=args.solver)
        return cnf

    def forward(self, x, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        for i in inds:
            x, logpx, reg_states = self.chain[i](x, logpx, reverse=reverse)

        return x, logpx, reg_states, self.reg_coeffs


class RFN(nn.Module):
    def __init__(self, args, device):
        super(RFN, self).__init__()
        self.args = args
        self.device = device
        self.data_type = args.type
        self.input_dim = args.input_dim
        self.memory_dim = args.memory_dim
        self.marginal = args.marginal
        self.softplus = nn.Softplus()

        self.norm_func = nn.Sequential(
            torch.nn.Linear(args.input_dim * args.memory_dim, args.input_dim * args.memory_dim),
            torch.nn.Softplus(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(args.input_dim * args.memory_dim, 2 * args.input_dim),
        )

        # construct marginal block
        if self.marginal == 'GRUODE':
            self.mgn_RFN = mgn_GRUODE(args, device)
        elif self.marginal == 'ODELSTM':
            self.mgn_RFN = mgn_ODELSTM(args, device)
        elif self.marginal == 'ODERNN':
            self.mgn_RFN = mgn_ODERNN(args, device)
        elif self.marginal == 'GRU-D':
            self.mgn_RFN = mgn_GRU_D(args, device)

        # construct dependence/joint block
        self.joint_RFN = joint_RFN(args)

        # initialize the classifier or p_model
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def base_dist(self, mean, var):
        return Normal(mean, var)

    def compute_loss(self, z_, delta_logp, params, reg_states, reg_coeffs):
        z, m = z_[:, :self.input_dim], z_[:, self.input_dim:2*self.input_dim]
        mean, var = torch.chunk(params, 2, dim=1)
        var = self.softplus(var)

        logpz = torch.sum(m * self.base_dist(mean, var).log_prob(z), dim=1, keepdim=True)
        logpx = logpz - delta_logp
        # nll = -torch.mean(logpx)
        nll = - torch.sum(logpx)/torch.sum(m)

        reg_states = tuple(torch.mean(rs) for rs in reg_states)
        if reg_coeffs:
            reg_loss = sum(
                reg_state * coeff for reg_state, coeff in zip(reg_states, reg_coeffs) if coeff != 0
            )
        return nll, reg_loss

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, device, dt=0.01, viz=False, val=False, t_corr=0.5):

        '''
        :param obs_times: the total observed times in the multivariate time series
        :param event_pt: the number of events at each time step
        :param sample_idx: the reindex of samples in one batch
        :param X: data of multivariable time series
        :param M: mask for each time series with 0 for no unobserved and 1 for observed
        :param batch_idx: the origin index of samples
        :param device: cpu or gpu
        :param T: the final time for multivariate time series
        :param dt: the time step of integral
        :return: loss of negative log likelihood
        '''

        current_time = 0.0

        mgn_h = torch.zeros([sample_idx.shape[0], self.input_dim * self.memory_dim]).to(device)
        if self.marginal == 'ODELSTM':
            mgn_c = torch.zeros([sample_idx.shape[0], self.input_dim * self.memory_dim]).to(device)
        if self.marginal == 'GRU-D':
            last_t = torch.zeros([sample_idx.shape[0], self.input_dim]).to(device)
            last_x = torch.zeros([sample_idx.shape[0], self.input_dim]).to(device)

        if val:
            num_sampling_samples = 100
            crps_list, crps_sum_list = [], []
            ND_err, ND_all = 0, 0
            if viz:
                sns.set_theme()
                sns.set_theme()
                plt.figure(figsize=[5 * len(t_corr), 5])
                plot_idx = 1

        para_list = []
        X_obs_list = []
        M_obs_list = []
        h_obs_list = []

        for j, time in enumerate(obs_times):
            # if no observation at current time, using ODE function updates the hidden state
            while current_time < time:
                # integral and update the marginal hidden state
                if self.marginal == 'ODELSTM':
                    mgn_h, mgn_c = self.mgn_RFN(current_time, (mgn_h, mgn_c), dt)
                elif self.marginal == 'GRUODE' or self.marginal == 'ODERNN':
                    mgn_h = self.mgn_RFN(current_time, mgn_h, dt)
                elif self.marginal == 'GRU-D' or self.marginal == 'GRU-delta-t':
                    pass
                current_time = current_time + dt

            # update the hidden state when observations exists
            if current_time >= time:
                # get samples which have new observations at current_time
                start = event_pt[j]
                end = event_pt[j + 1]
                X_obs = X[start:end]
                M_obs = M[start:end]
                i_obs = batch_idx[start:end].type(torch.LongTensor)

                if val:
                    h_obs_list_val = mgn_h[i_obs].repeat_interleave(num_sampling_samples, dim=0)
                    M_obs_list_val = M_obs.repeat_interleave(num_sampling_samples, dim=0)
                    mean, var = torch.chunk(self.norm_func(mgn_h[i_obs]), 2, dim=1)
                    base_sample = self.base_dist(mean, self.softplus(var))
                    z = base_sample.sample(sample_shape=torch.Size([num_sampling_samples])).type(torch.float32).to(device).permute(1, 0, 2)
                    z_ = z.reshape(z.shape[0] * z.shape[1], z.shape[-1])
                    logpz = torch.zeros(z_.shape[0], 1).to(z_)

                    x, _, _, _ = self.joint_RFN(torch.cat([z_, M_obs_list_val, h_obs_list_val], dim=1), logpz, reverse=True)
                    x = x[:, :self.input_dim]
                    # the shape of x is [num_sampling_samples, batch_size_at_this_time, num_variables]
                    x = x.reshape([X_obs.shape[0], num_sampling_samples, X_obs.shape[-1]]).permute(1, 0, 2)
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
                            utils.viz_flow(np.round(current_time, 2), x)
                            plot_idx += 1

                '''compute the point estimation error'''
                # update h in marginal component
                h_obs_list.append(mgn_h[i_obs])
                para_list.append(self.norm_func(mgn_h[i_obs]))
                # update the hidden state
                if self.marginal == 'ODELSTM':
                    mgn_h, mgn_c = self.mgn_RFN(current_time, (mgn_h, mgn_c), dt, X_obs, M_obs, i_obs, update=True)
                elif self.marginal == 'GRUODE' or self.marginal == 'ODERNN':
                    mgn_h = self.mgn_RFN(current_time, mgn_h, dt, X_obs, M_obs, i_obs, update=True)
                elif self.marginal == 'GRU-D':
                    mgn_h, last_x, last_t = self.mgn_RFN(current_time, mgn_h, X_obs, M_obs, i_obs, last_x, last_t)

                X_obs_list.append(X_obs)
                M_obs_list.append(M_obs)

        X_obs_list = torch.cat(X_obs_list, dim=0)
        M_obs_list = torch.cat(M_obs_list, dim=0)
        h_obs_list = torch.cat(h_obs_list, dim=0)
        para_list = torch.cat(para_list, dim=0)
        # compute negative log likelihood via CNF for interval estimation
        logpx = torch.zeros(X_obs_list.shape[0], 1).to(X_obs_list)
        z, delta_logp, reg_states, reg_coeffs = self.joint_RFN(torch.cat([X_obs_list, M_obs_list, h_obs_list], dim=1), logpx)
        loss_nll, loss_reg = self.compute_loss(z, delta_logp, para_list, reg_states, reg_coeffs)

        if val:
            return loss_nll, np.mean(list(chain(*crps_list))), np.mean(list(chain(*crps_sum_list))), ND_err/ND_all
        else:
            return loss_nll, loss_reg

    def predict(self, obs_times, event_pt, sample_idx, X, M, batch_idx, device, T=1, dt=0.01):
        current_time = 0
        mgn_h = torch.zeros([sample_idx.shape[0], self.input_dim * self.memory_dim]).to(device)
        num_sampling_samples = 100
        X_gen_list = []
        t_list = []
        obs_times = np.append(obs_times, 0.99)
        for j, time in enumerate(obs_times):
            while current_time < time:
                # integral and update the marginal hidden state
                if self.marginal == 'ODELSTM':
                    mgn_h, mgn_c = self.mgn_RFN(current_time, (mgn_h, mgn_c), dt)
                elif self.marginal == 'GRUODE' or self.marginal == 'ODERNN':
                    mgn_h = self.mgn_RFN(current_time, mgn_h, dt)
                elif self.marginal == 'GRU-D' or self.marginal == 'GRU-delta-t':
                    pass
                current_time = current_time + dt

                mean, var = torch.chunk(self.norm_func(mgn_h), 2, dim=1)
                base_sample = self.base_dist(mean, self.softplus(var))

                z = base_sample.sample(sample_shape=torch.Size([num_sampling_samples])).type(torch.float32).to(device).permute(1, 0, 2)
                z_ = z.reshape(z.shape[0] * z.shape[1], z.shape[-1])
                logpz = torch.zeros(z_.shape[0], 1).to(z_)
                t_obs = torch.ones(z_.shape[0], 1).to(z_) * current_time
                X_gen, _, _, _ = self.joint_RFN(torch.cat([z_, t_obs], dim=1), logpz, reverse=True)
                # the shape of x is [num_sampling_samples, batch_size_at_this_time, num_variables]
                X_gen = X_gen[:, :-1].reshape([sample_idx.shape[0], num_sampling_samples, self.input_dim]).permute(1, 0, 2)

                X_gen_list.append(X_gen.squeeze().detach().cpu().numpy())
                t_list.append(current_time)

            # update the hidden state when observations exists
            try:
                if current_time >= time:
                    start = event_pt[j]
                    end = event_pt[j + 1]
                    X_obs = X[start:end]
                    M_obs = M[start:end]
                    i_obs = batch_idx[start:end].type(torch.LongTensor)
                    if self.marginal == 'ODELSTM':
                        mgn_h, mgn_c = self.mgn_RFN(current_time, (mgn_h, mgn_c), dt, X_obs, M_obs, i_obs, update=True)
                    elif self.marginal == 'GRUODE' or self.marginal == 'ODERNN' or self.marginal == 'GRU-delta-t':
                        mgn_h = self.mgn_RFN(current_time, mgn_h, dt, X_obs, M_obs, i_obs, update=True)
                    elif self.marginal == 'GRU-D':
                        mgn_h, last_x, last_t = self.mgn_RFN(current_time, mgn_h, X_obs, M_obs, i_obs, last_x, last_t)
            except:
                pass

        return t_list, X_gen_list


