import copy
import numpy as np
import random
from utils.func import MapInfo


class Space:
    def __init__(self, map_info: MapInfo, n_agents):
        self.map = map_info
        self.n_agents = n_agents
        self.curr = [None] * n_agents

    def update(self, agent_i, pos):
        self.curr[agent_i] = pos

    def sample(self):
        act_n = []
        act_idx_n = []
        for agent_i in range(self.n_agents):
            acts = self.map.get_edges(self.curr[agent_i])
            act_idx = np.random.randint(len(acts))
            act = acts[act_idx]
            act_idx_n.append(act_idx)
            act_n.append(act)
        return act_n, act_idx_n


class Env:
    def __init__(self, map_info, full_observation: bool = False, n_agents=2, step_cost=-1, max_steps=100, init_pos=1,
                 view_target: bool = False, view_step: bool = False, view_intarget: bool=False, is_onehot: bool = False,
                 target_pos_ll=None):
        assert 1 <= n_agents <= 5
        self.map_info = map_info
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.full_observation = full_observation
        self.is_onehot = is_onehot
        self.view_target = view_target
        self.view_step = view_step
        self.view_intarget = view_intarget
        self.step_cnt = 0
        self.total_reward = None
        self.action_space = Space(map_info, n_agents)
        self.agent_dones = [False for _ in range(self.n_agents)]
        init_agent_pos = {
            0: init_pos,
            1: init_pos,
            2: init_pos,
            3: init_pos,
            4: init_pos,
        }
        self.init_agent_pos = {}
        self.target_pos_ll = [self.map_info.n_node]
        if target_pos_ll:
            self.target_pos_ll = target_pos_ll
        self.init_target_pos = random.choice(self.target_pos_ll)
        for agent_i in range(n_agents):
            self.init_agent_pos[agent_i] = init_agent_pos[agent_i]

        self.__init_full_obs()

    def __init_full_obs(self):
        self.agents_pos = copy.copy(self.init_agent_pos)
        self.init_target_pos = random.choice(self.target_pos_ll)#self.map_info.n_node #np.random.randint(1, self.map_info.n_node+1)
        # while self.init_target_pos == self.init_agent_pos[0]:
        #     self.init_target_pos = np.random.randint(1, self.map_info.n_node + 1)
        self.target_pos = self.init_target_pos  # np.random.randint(1, self.map_info.n_node+1)

    def __is_agent_done(self, agent_i):
        return self.target_pos == self.agents_pos[agent_i]

    def __target_move(self):
        if np.random.random() <= 0.2:
            self.target_pos = np.random.choice(self.map_info.get_next_nodes(self.target_pos))

    def __update_agent_pos(self, agent_i, n_pos):
        parti_tmp = self.map_info.get_next_nodes(self.agents_pos[agent_i])
        if n_pos not in parti_tmp:
            raise Exception('Edge[{}, {}] Not Found!'.format(self.agents_pos[agent_i]), n_pos)
        self.agents_pos[agent_i] = n_pos

    def get_agent_obs(self):
        obs = []
        for agent_i in range(self.n_agents):
            pos = self.agents_pos[agent_i]
            agent_i_obs = self.map_info.get_edges(pos)
            if self.is_onehot:
                agent_i_obs = list(map(lambda x: [np.eye(self.map_info.n_node)[x[0]-1],
                                                  np.eye(self.map_info.n_node)[x[1]-1]], agent_i_obs))
            if self.view_target:
                if self.is_onehot:
                    agent_i_obs = list(map(lambda x: x + [np.eye(self.map_info.n_node)[self.target_pos-1]], agent_i_obs))
                else:
                    agent_i_obs = list(map(lambda x: x + [self.target_pos], agent_i_obs))
            # agent_i_obs = list(map(lambda x: np.array(x).flatten(), agent_i_obs))
            if self.view_intarget:
                if self.is_onehot:
                    agent_i_obs = list(map(lambda x: x + [np.eye(self.map_info.n_node)[self.init_target_pos-1]], agent_i_obs))
                else:
                    agent_i_obs = list(map(lambda x: x + [self.init_target_pos], agent_i_obs))
            agent_i_obs = list(map(lambda x: np.array(x).flatten(), agent_i_obs))

            if self.view_step:
                agent_i_obs = list(map(lambda x: x.tolist() + [self.step_cnt], agent_i_obs))
                agent_i_obs = list(map(lambda x: np.array(x), agent_i_obs))
            obs.append(agent_i_obs)
        if self.full_observation:
            obs = np.array(obs).flatten()
            obs = [obs for _ in range(self.n_agents)]
        return obs

    def step(self, agents_action):
        self.__target_move()
        self.step_cnt += 1
        rewards = [self.map_info.mu[self.agents_pos[agent_i]] if self.step_cost == "weight" else self.step_cost for
                   agent_i in range(self.n_agents)]
        done_flag = False
        for agent_i, action in enumerate(agents_action):
            act = action[1]
            if self.is_onehot:
                act = action[self.map_info.n_node:].nonzero()[0][0] + 1
            self.__update_agent_pos(agent_i, act)
            self.action_space.update(agent_i, self.agents_pos[agent_i])
            self.agent_dones[agent_i] = self.__is_agent_done(agent_i)
            if self.agent_dones[agent_i] and not done_flag:
                done_flag = True
                rewards[agent_i] = 5

        if self.step_cnt >= self.max_steps:
            self.agent_dones = [True] * self.n_agents

        for agent_i in range(self.n_agents):
            self.total_reward[agent_i] += rewards[agent_i]

        return self.get_agent_obs(), rewards, self.agent_dones, {}

    def reset(self):
        self.__init_full_obs()
        self.step_cnt = 0
        self.agent_dones = [False for _ in range(self.n_agents)]
        self.total_reward = [0 for _ in range(self.n_agents)]
        for agent_i in range(self.n_agents):
            self.action_space.update(agent_i, self.agents_pos[agent_i])
        return self.get_agent_obs()

    def render(self, mode='human'):
        if mode == "human":
            print("\r  step_count:{};   agents_pos:{};  target_pos:{};  total_rewards:{}".
                  format(self.step_cnt, self.agents_pos, self.target_pos, self.total_reward))

    def seed(self, n=None):
        if not n:
            np.random.seed(n)
        return [n]

    def close(self):
        del self

# map1 = MapInfo()
# env = Env(map1)
# env.reset()
