import torch
import random
from ._base import _Base
from utils import ReplayMemory, DQRN_Transition, PrioritizedReplayMemory, soft_update, hard_update, networks
from utils import LinearDecay
import copy
import numpy as np
import torch.nn.functional as F
import time
import pandas as pd


class C51_RNN(_Base):
    """
    Value Decomposition Network + Implicit Quantile + Double DQN + Prioritized Replay + Soft Target Updates

    Paper: https://arxiv.org/pdf/1706.05296.pdf
    """
    def __init__(self, env_fn, model_fn, mixer_fn, lr, discount, batch_size, device, mem_len, tau, train_episodes,
                 episode_max_steps, path):
        super().__init__(env_fn, model_fn, mixer_fn, lr, discount, batch_size, device, train_episodes,
                         episode_max_steps, path)
        self.memory = PrioritizedReplayMemory(mem_len, with_hidden=True)
        self.tau = tau

        self.target_model = model_fn().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.target_mixer = copy.deepcopy(self.mixer)
        self.target_mixer.eval()

        self.exploration = LinearDecay(0.1, 1.0, self.train_episodes)
        self._update_iter = 0

    def projection_distribution(self, next_dist, rewards, dones):
        batch_size = next_dist.shape[0]
        V_max = self.model.V_max
        V_min = self.model.V_min
        n_atoms = self.model.n_atoms
        delta_z = float(V_max - V_min) / (n_atoms - 1)

        rewards = rewards.expand_as(next_dist).to(self.device)
        dones = dones.expand_as(next_dist).to(self.device)
        support = self.model.support.unsqueeze(0).expand_as(next_dist).to(self.device)

        Tz = rewards + (1 - dones) * support
        Tz = Tz.clamp(min=V_min, max=V_max)
        bj = (Tz - V_min) / delta_z
        m_l = bj.floor().long()
        m_u = bj.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * n_atoms, batch_size).long() \
            .unsqueeze(1).expand(batch_size, n_atoms).to(self.device)

        m_prob = torch.zeros(next_dist.size()).to(self.device)
        m_prob.view(-1).index_add_(0, (m_l + offset).view(-1), (next_dist * (m_u.float() - bj)).view(-1))
        m_prob.view(-1).index_add_(0, (m_u + offset).view(-1), (next_dist * (bj - m_l.float())).view(-1))

        return m_prob

    def _update(self):
        self.model.train()
        if self.batch_size > len(self.memory):
            self.model.eval()
            return None

        # Todo: move this beta in the Prioritized Replay memory
        beta_start = 0.4
        beta = min(1.0, beta_start + (self._update_iter + 1) * (1.0 - beta_start) / 5000)

        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        batch = DQRN_Transition(*zip(*transitions))

        # obs_batch = torch.FloatTensor(list(batch.state)).to(self.device)
        action_batch = torch.FloatTensor(np.array(batch.action)).view(self.batch_size, self.env.n_agents, -1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).view(self.batch_size, self.env.n_agents, -1).to(self.device)
        # next_obs_batch = torch.FloatTensor(list(batch.next_state)).to(self.device)
        next_obs_batch = np.array(batch.next_state, dtype=object).reshape(
            (self.batch_size, self.env.n_agents, -1))
        hidden_batch = torch.cat(batch.hidden).view(self.batch_size, self.env.n_agents, -1).to(self.device)
        next_hidden_batch = torch.cat(batch.next_hidden).view(self.batch_size, self.env.n_agents, -1).to(
            self.device)
        dones_bath = torch.FloatTensor(batch.done).reshape((self.batch_size, -1))

        # calc loss
        # overall_pred_q, target_q = 0, 0
        q_val, target_q_val = [], []
        for agent_i in range(self.model.n_agents):
            q_val_i = self.model.agent(agent_i).cal_feature(action_batch[:, agent_i], hidden_batch[:, agent_i].unsqueeze(0))[0].view(self.batch_size, 1, -1)
            try:
                q_val = torch.cat((q_val, q_val_i), dim=-2)
            except:
                q_val = q_val_i

            target_next_obs_q = [torch.zeros(q_val_i[0].shape).to(self.device)] * q_val_i.shape[0]
            # Double DQN update
            for j, _next_obs in enumerate(next_obs_batch[:, agent_i]):
                _next_obs = _next_obs[0]
                if not batch.done[j]:
                    torch_next_obs = torch.FloatTensor(np.array(_next_obs)).to(self.device)
                    _qs, _ = self.model.agent(agent_i).\
                        cal_feature(torch_next_obs, next_hidden_batch[j, agent_i, :].expand((1, torch_next_obs.shape[0], 128)))
                    _max_idx = torch.sum(torch.multiply(F.softmax(_qs, dim=-1), self.model.support), dim=1).argmax(0).item()
                    _max_q = _qs[_max_idx]
                    target_next_obs_q[j] = _max_q.unsqueeze(0)
            target_next_obs_q = torch.cat(target_next_obs_q).unsqueeze(1)
            try:
                target_q_val = torch.cat((target_q_val, target_next_obs_q.detach()), dim=1)
            except:
                target_q_val = target_next_obs_q.detach()

        # Mix
        if self.env.n_agents > 1:
            overall_pred_q = self.mixer(q_val, action_batch, hidden_batch)
            target_q = self.target_mixer(target_q_val, action_batch, next_hidden_batch)
        else:
            overall_pred_q = F.softmax(q_val).squeeze(1)
            target_q = F.softmax(target_q_val).squeeze(1)

        target_q = self.projection_distribution(target_q, (reward_batch.sum(1))/self.model.n_agents, dones_bath)
        loss = - (target_q * overall_pred_q.log()).sum(1)
        prios = loss + 1e-5
        loss = loss.mean(dim=0)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        # if total_norm.isinf() or total_norm.isnan() or torch.isclose(total_norm, torch.zeros_like(total_norm)):
        #     print(f"{total_norm}")
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.memory.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        # update target network
        # Todo: Make 100 as a parameter
        # if self.__update_iter % 100:
        #     hard_update(self.target_model, self.model)
        soft_update(self.target_model, self.model, self.tau)

        # log
        self.writer.add_scalar('_overall/critic_loss', loss, self._step_iter)
        self.writer.add_scalar('_overall/beta', beta, self._step_iter)

        # just keep track of update counts
        self._update_iter += 1

        # resuming the model in eval mode
        self.model.eval()

        return loss.item()

    def _select_action(self, model, obs_n, hiddens, explore=False):
        """ selects epsilon greedy action for the state """
        act_n = []
        _hiddens_n = [[] for _ in range(self.model.n_agents)]
        n_hiddens = torch.zeros(hiddens.shape).to(self.device)
        for agent_i in range(self.model.n_agents):
            _qs = None
            for obs in obs_n[agent_i]:
                torch_obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                _q, _hidden = model.agent(agent_i)(torch_obs, hiddens[agent_i])
                try:
                    _qs = torch.cat((_qs, _q), dim=0)
                except:
                    _qs = _q
                _hiddens_n[agent_i].append(_hidden)
            _max_idx = torch.sum(torch.multiply(F.softmax(_qs, dim=-1), self.model.support), dim=-1).argmax(0).item()
            act_n.append(obs_n[agent_i][_max_idx])
            n_hiddens[agent_i] = _hiddens_n[agent_i][_max_idx]
        if explore and self.exploration.eps > np.random.random():
            act_n = []
            act_idx_n = self.env.action_space.sample()[1]
            for agent_i in range(self.model.n_agents):
                act_n.append(obs_n[agent_i][act_idx_n[agent_i]])
                n_hiddens[agent_i] = _hiddens_n[agent_i][act_idx_n[agent_i]]
        return act_n, n_hiddens

    def _train(self, episodes):
        self.model.eval()
        train_rewards = []
        train_loss = []

        for ep in range(episodes):
            terminal = False
            obs_n = self.env.reset()
            ep_step = 0
            ep_reward = [0 for _ in range(self.model.n_agents)]
            hiddens = torch.zeros(self.env.n_agents, 1, 1, 128, dtype=torch.float).to(self.device)
            while not terminal:
                action_n, next_hiddens = self._select_action(self.model, obs_n, hiddens=hiddens, explore=True)
                next_obs_n, reward_n, done_n, info = self.env.step(action_n)
                terminal = any(done_n) or ep_step >= self.episode_max_steps
                self.memory.push(obs_n, action_n, next_obs_n, hiddens.detach(), next_hiddens.detach(), reward_n, terminal)
                loss = self._update()
                obs_n = next_obs_n
                hiddens = next_hiddens
                ep_step += 1
                self._step_iter += 1
                if loss is not None:
                    train_loss.append(loss)
                for i, r_n in enumerate(reward_n):
                    ep_reward[i] += r_n
            train_rewards.append(ep_reward)
            self.exploration.update()

            # log - training
            for i, r_n in enumerate(ep_reward):
                self.writer.add_scalar('agent_{}/train_reward'.format(i), r_n, self._step_iter)
            self.writer.add_scalar('_overall/train_reward', sum(ep_reward), self._step_iter)
            self.writer.add_scalar('_overall/train_ep_steps', ep_step, self._step_iter)
            self.writer.add_scalar('_overall/exploration_rate', self.exploration.eps, self._step_iter)

            print(ep, sum(ep_reward))

        return np.array(train_rewards).mean(axis=0), (np.mean(train_loss) if len(train_loss) > 0 else [])

    def test(self, episodes, render=False, log=False, record=False):
        self.model.eval()
        env = self.env
        # if record:
        #     env = Monitor(self.env_fn(), directory=os.path.join(self.path, 'recordings'), force=True,
        #                   video_callable=lambda episode_id: True)
        with torch.no_grad():
            test_rewards = []
            total_test_steps=0
            for ep in range(episodes):
                terminal = False
                obs_n = env.reset()
                if render:
                    env.render()
                step = 0
                ep_reward = [0 for _ in range(self.env.n_agents)]
                t0 = time.perf_counter()
                hiddens = torch.zeros(self.env.n_agents, 1, 1, 128, dtype=torch.float).to(self.device)
                while not terminal:
                    action_n, hiddens = self._select_action(self.model, obs_n, hiddens=hiddens, explore=False)
                    next_obs_n, reward_n, done_n, info = env.step(action_n)
                    terminal = any(done_n) or step >= self.episode_max_steps
                    if render:
                        env.render()
                    obs_n = next_obs_n
                    step += 1
                    for i, r_n in enumerate(reward_n):
                        ep_reward[i] += r_n

                if record:
                    try:
                        pf_list.append({"target start vertex": self.env.init_target_pos,
                                    "target capture vertex": self.env.target_pos,
                                    "total capture steps": self.env.step_cnt,
                                    "total planning time": time.perf_counter() - t0})
                    except:
                        pf_list = []
                        pf_list.append({"target start vertex": self.env.init_target_pos,
                                    "target capture vertex": self.env.target_pos,
                                    "total capture steps": self.env.step_cnt,
                                    "total planning time": time.perf_counter() - t0})

                total_test_steps += step
                test_rewards.append(ep_reward)

            test_rewards = np.array(test_rewards).mean(axis=0)
            if log:
                # log - test
                for i, r_n in enumerate(test_rewards):
                    self.writer.add_scalar('agent_{}/eval_reward'.format(i), r_n, self._step_iter)
                self.writer.add_scalar('_overall/eval_reward', sum(test_rewards), self._step_iter)
                self.writer.add_scalar('_overall/test_ep_steps', total_test_steps / episodes, self._step_iter)
        if record:
            pf = pd.DataFrame(pf_list)
            pf.to_excel(excel_writer=self.path + '.xlsx', sheet_name='sheet_1')

        return test_rewards
