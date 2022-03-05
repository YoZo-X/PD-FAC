import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


# ****************************************************************
# DQN Network
# ****************************************************************

class DQNAgent(nn.Module):
    def __init__(self, input_size):
        super(DQNAgent, self).__init__()

        # self.action_space = action_space
        # self.normalize_size = normalize_size
        self.seq = nn.Sequential(nn.Linear(input_size, 64),
                                 nn.ReLU(),
                                 # NoisyLinear(64, 1))
                                 nn.Linear(64, 1))

        for m in self.seq:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.seq(x)


class DQNNet(nn.Module):
    def __init__(self, env, seed):
        super(DQNNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        input_size = len(env.get_agent_obs()[0][0])
        self.n_agents = env.n_agents
        for i in range(self.n_agents):
            agent_i = 'agent_{}'.format(i)
            setattr(self, agent_i, DQNAgent(input_size=input_size))

    def agent(self, i):
        return getattr(self, 'agent_{}'.format(i))


# ****************************************************************
# DDPG Network
# ****************************************************************

# class ActorAgent(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#
#         # self.action_space = action_space
#         self.seq = nn.Sequential(nn.Linear(input_size, 64),
#                                      nn.ReLU(),
#                                      nn.Linear(64, output_size),
#                                  nn.Softmax())
#
#         for m in self.seq:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         return self.seq(x)
#
#
# class CriticAgent(nn.Module):
#     def __init__(self, state_size, actor_size):
#         super().__init__()
#         self.seq = nn.Sequential(nn.Linear(state_size+actor_size, 128),
#                                      nn.ReLU(),
#                                      nn.Linear(128, 1))
#         for m in self.seq:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x, u):
#         x = torch.cat((u, x), dim=1)
#         return self.seq(x)
#
#
# class DDPGCritic(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.n_agents = env.n_agents
#         state_size = len(env.get_agent_obs()[0][0])
#         actor_size = env.map_info.n_node
#         for i in range(self.n_agents):
#             agent_i = 'agent_{}'.format(i)
#             setattr(self, agent_i, CriticAgent(state_size, actor_size))
#
#     def forward(self, i):
#         return getattr(self, 'agent_{}'.format(i))
#
#
# class DDPGActor(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.n_agents = env.n_agents
#         input_size = len(env.get_agent_obs()[0][0])
#         output_size = env.map_info.n_node
#         for i in range(self.n_agents):
#             agent_i = 'agent_{}'.format(i)
#             setattr(self, agent_i, ActorAgent(input_size, output_size))
#
#     def forward(self, i):
#         return getattr(self, 'agent_{}'.format(i))
#
#
# class MADDPGNet(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.n_agents = env.n_agents
#         self.actor = DDPGActor(env)
#         self.critic = DDPGCritic(env)


# ****************************************************************
# IQN Network
# ****************************************************************

def calc_cos(batch_size, n_cos=64, n_tau=32):
    """
    Calculating the cosinus values depending on the number of tau samples
    """
    pis = torch.FloatTensor([np.pi * i for i in range(n_cos)]).view(1, 1, n_cos)
    taus = torch.rand(batch_size, n_tau).unsqueeze(-1)  # (batch_size, n_tau, 1)
    cos = torch.cos(taus * pis)

    assert cos.shape == (batch_size, n_tau, n_cos), "cos shape is incorrect"
    return cos, taus


class QuantileAgent(nn.Module):
    def __init__(self, state_size, action_size, layer_size, n_cos=64, n_tau=8):
        super(QuantileAgent, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.n_tau = n_tau
        self.n_cos = n_cos
        self.layer_size = layer_size
        # Starting from 0 as in the paper

        self.head = nn.Linear(self.input_shape, layer_size)
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)

    def forward(self, states, cos):
        """
        Quantile Calculation depending on the number of tau

        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]

        """
        batch_size = states.shape[0]
        assert cos.shape == (batch_size, self.n_tau, self.n_cos), "cos shape is incorrect"
        x = torch.relu(self.head(states))
        # cos, taus = calc_cos(batch_size, num_tau)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * self.n_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, self.n_tau,
                                                         self.layer_size)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * self.n_tau, self.layer_size)

        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)

        return out.view(batch_size, self.action_size, self.n_tau)


class IQNNet(nn.Module):
    def __init__(self, env, seed, n_cos=64, n_tau=8):
        super(IQNNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        state_size = len(env.get_agent_obs()[0][0])
        self.n_agents = env.n_agents
        self.n_cos = n_cos
        self.n_taus = n_tau
        for i in range(self.n_agents):
            agent_i = 'agent_{}'.format(i)
            setattr(self, agent_i, QuantileAgent(state_size, 1, 128, n_cos, n_tau))

    def calculate_huber_loss(self, td_errors, k=1.0):
        """
        Calculate huber loss element-wisely depending on kappa k.
        """
        loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
        assert loss.shape == (td_errors.shape[0], self.n_taus, self.n_taus), "huber loss has wrong shape"
        return loss

    def agent(self, i, states, cos):
        agent = getattr(self, 'agent_{}'.format(i))
        return agent(states, cos)


# ****************************************************************
# C51 Network
# ****************************************************************

class C51Agent(nn.Module):
    def __init__(self, state_size, layer_size, n_atoms):
        super(C51Agent, self).__init__()
        self.input_shape = state_size
        self.output_size = n_atoms
        self.layer_size = layer_size
        self.seq = nn.Sequential(nn.Linear(state_size, layer_size),
                                 nn.LeakyReLU(),
                                 nn.Linear(layer_size, n_atoms),
                                 # NoisyLinear(layer_size, n_atoms, 0.1),
                                 )

        for m in self.seq:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return F.softmax(self.seq(x))


class C51Net(nn.Module):
    def __init__(self, env, seed, n_atoms=51, V_min=-20, V_max=20, device='cpu'):
        super(C51Net, self).__init__()
        self.seed = torch.manual_seed(seed)
        state_size = len(env.get_agent_obs()[0][0])
        self.n_agents = env.n_agents
        self.n_atoms = n_atoms
        self.V_min = V_min
        self.V_max = V_max
        self.support = torch.linspace(V_min, V_max, n_atoms).to(device)
        for i in range(self.n_agents):
            agent_i = 'agent_{}'.format(i)
            setattr(self, agent_i, C51Agent(state_size, 128, n_atoms))

    def agent(self, i):
        return getattr(self, 'agent_{}'.format(i))


# ****************************************************************
# DQRN Network
# ****************************************************************

class DQRNAgent(nn.Module):
    def __init__(self, input_size, device="cpu"):
        super(DQRNAgent, self).__init__()

        # self.action_space = action_space
        # self.normalize_size = normalize_size
        self.device = device
        self.seq = nn.Sequential(nn.Linear(input_size, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU())
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.ff = nn.Linear(128, 1)

        for m in self.seq:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.ff.weight)
        nn.init.constant_(self.ff.bias, 0)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 128, dtype=torch.float).to(self.device)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]
        if len(x.shape) == 3:
            len_seq = x.shape[1]
            x = self.seq(x).view(batch_size, len_seq, -1)
        else:
            x = self.seq(x).view(batch_size, 1, -1)
        hidden = self.init_hidden(batch_size) if hidden is None else hidden
        xx, n_hidden = self.gru(x, hidden)
        # x = self.ff(x)
        # x = torch.cat((x, F.relu(xx)), -1)
        x = self.ff(F.relu(xx))
        return x, n_hidden


class DQRNNet(nn.Module):
    def __init__(self, env, seed, hidden_layer=128, device="cpu"):
        super(DQRNNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_layer = hidden_layer
        input_size = len(env.get_agent_obs()[0][0])
        self.n_agents = env.n_agents
        for i in range(self.n_agents):
            agent_i = 'agent_{}'.format(i)
            setattr(self, agent_i, DQRNAgent(input_size=input_size, device=device))

    def agent(self, i):
        return getattr(self, 'agent_{}'.format(i))


# ****************************************************************
# C51RNN Network
# ****************************************************************

class C51RNNAgent(nn.Module):
    def __init__(self, state_size, n_atoms=51, device='cpu'):
        super(C51RNNAgent, self).__init__()
        self.device = device
        self.input_shape = state_size
        self.output_size = n_atoms

        self.seq = nn.Sequential(nn.Linear(state_size, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU())
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.ff = nn.Linear(128, self.output_size)
        for m in self.seq:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.ff.weight)
        nn.init.constant_(self.ff.bias, 0)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 128, dtype=torch.float).to(self.device)

    def cal_feature(self, x, hidden=None):
        batch_size = x.shape[0]
        if len(x.shape) == 3:
            len_seq = x.shape[1]
            x = self.seq(x).view(batch_size, len_seq, -1)
        else:
            x = self.seq(x).view(batch_size, 1, -1)
        hidden = self.init_hidden(batch_size) if hidden is None else hidden
        xx, n_hidden = self.gru(x, hidden.contiguous())
        x = self.ff(F.relu(xx))
        return x.squeeze(1), n_hidden

    def forward(self, x, hidden=None):
        x, n_hidden = self.cal_feature(x, hidden)
        return F.softmax(x, dim=-1), n_hidden


class C51RNNNet(nn.Module):
    def __init__(self, env, seed, n_atoms=51, V_min=-20, V_max=20, device='cpu'):
        super(C51RNNNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        state_size = len(env.get_agent_obs()[0][0])
        self.n_agents = env.n_agents
        self.n_atoms = n_atoms
        self.V_min = V_min
        self.V_max = V_max
        self.support = torch.linspace(V_min, V_max, n_atoms).to(device)
        for i in range(self.n_agents):
            agent_i = 'agent_{}'.format(i)
            setattr(self, agent_i, C51RNNAgent(state_size, n_atoms, device=device))

    def agent(self, i):
        return getattr(self, 'agent_{}'.format(i))


# ****************************************************************
# IQN_RNN Network
# ****************************************************************

class QuantileRNNAgent(nn.Module):
    def __init__(self, state_size, action_size, n_cos=64, n_tau=8, device='cpu'):
        super(QuantileRNNAgent, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.n_tau = n_tau
        self.n_cos = n_cos
        self.device = device

        self.cos_embedding = nn.Linear(self.n_cos, 128)
        self.ff = nn.Linear(128, 128)
        self.ff_1 = nn.Linear(128, 128)
        self.ff_2 = nn.Linear(128, action_size)

        self.head = nn.Sequential(nn.Linear(self.input_shape, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 128),
                                  nn.ReLU())
        self.gru = nn.GRU(128, 128, batch_first=True)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.ff.weight)
        nn.init.constant_(self.ff.bias, 0)
        nn.init.xavier_uniform_(self.ff_1.weight)
        nn.init.constant_(self.ff.bias, 0)
        nn.init.xavier_uniform_(self.ff_2.weight)
        nn.init.constant_(self.ff.bias, 0)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 128, dtype=torch.float).to(self.device)

    def forward(self, states, cos, hidden=None):
        batch_size = states.shape[0]
        assert cos.shape == (batch_size, self.n_tau, self.n_cos), "cos shape is incorrect"
        x = torch.relu(self.head(states)).view(batch_size, 1, -1)
        hidden = self.init_hidden(batch_size) if hidden is None else hidden
        xx, n_hidden = self.gru(x, hidden.contiguous())
        x = torch.relu(self.ff(xx)).squeeze(1)
        # cos, taus = calc_cos(batch_size, num_tau)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * self.n_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, self.n_tau,
                                                         128)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * self.n_tau, 128)

        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)

        return out.view(batch_size, self.action_size, self.n_tau), n_hidden


class IQNRNNNet(nn.Module):
    def __init__(self, env, seed, n_cos=64, n_tau=8, device='cpu'):
        super(IQNRNNNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        state_size = len(env.get_agent_obs()[0][0])
        self.n_agents = env.n_agents
        self.n_cos = n_cos
        self.n_taus = n_tau
        for i in range(self.n_agents):
            agent_i = 'agent_{}'.format(i)
            setattr(self, agent_i, QuantileRNNAgent(state_size, 1, n_cos, n_tau, device))

    def calculate_huber_loss(self, td_errors, k=1.0):
        """
        Calculate huber loss element-wisely depending on kappa k.
        """
        loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
        assert loss.shape == (td_errors.shape[0], self.n_taus, self.n_taus), "huber loss has wrong shape"
        return loss

    def agent(self, i, states, cos, hidden=None):
        agent = getattr(self, 'agent_{}'.format(i))
        return agent(states, cos, hidden)
