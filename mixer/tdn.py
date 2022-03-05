import torch
import torch.nn as nn
import torch.nn.functional as F


class TDN_Mixer(nn.Module):
    def __init__(self, env, n_atoms, is_simple=False):
        super(TDN_Mixer, self).__init__()
        self.n_agents = env.n_agents
        self.is_simple = is_simple
        self.state_dim = len(env.get_agent_obs()[0][0]) * env.n_agents
        self.n_atoms = n_atoms
        self.hyper1 = nn.Sequential(nn.Linear(self.state_dim, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, self.n_agents))
        self.hyper2 = nn.Sequential(nn.Linear(128 * env.n_agents, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, self.n_agents))
        for m in self.hyper1:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.hyper2:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, agent_qs, batch, hidden):
        if not self.is_simple:
            x = batch.reshape(-1, self.state_dim)
            xx = hidden.reshape(-1, 128 * self.n_agents)
            # tmp = torch.sum(agent_qs[0, 0])
            w = F.softmax(self.hyper1(x) + self.hyper2(xx), dim=-1).unsqueeze(-1)
        else:
            w = torch.ones((batch.shape[0], self.n_agents, 1)).to(agent_qs.device)/self.n_agents
        q_mixture = torch.sum(w * F.softmax(agent_qs, -1), -2)
        return q_mixture
