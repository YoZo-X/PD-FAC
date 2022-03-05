import numpy as np


class LinearDecay:
    """ Linearly Decays epsilon for exploration between a range of episodes"""

    def __init__(self, min_eps, max_eps, total_episodes):
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.total_episodes = total_episodes
        self.curr_episodes = 0
        # Todo: make 0.5 available as parameter
        self._threshold_episodes = 0.5 * total_episodes
        self.eps = max_eps

    def update(self):
        self.curr_episodes += 1
        eps = self.max_eps * (self._threshold_episodes - self.curr_episodes) / self._threshold_episodes
        self.eps = max(self.min_eps, eps)


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
