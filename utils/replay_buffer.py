from collections import namedtuple
import random
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

DQRN_Transition = namedtuple('Transition',
                             ('state', 'action', 'next_state', 'hidden', 'next_hidden', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, capacity, with_hidden=False):
        self.capacity = capacity
        self.Transition = Transition
        if with_hidden:
            self.Transition = DQRN_Transition
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):
    def __init__(self, capacity, prob_alpha=0.6, with_hidden=False):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.Transition = Transition
        if with_hidden:
            self.Transition = DQRN_Transition
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, *args):

        max_prio = np.max(self.priorities) if self.memory else 1.0

        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.priorities[self.position] = max_prio

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        probs = prios ** self.prob_alpha
        probs /= np.sum(probs)
        indices = np.random.choice(len(self.memory), batch_size, p=probs if np.nansum(probs) == 1 else None)
        batch = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= np.max(weights)
        weights = np.array(weights, dtype=np.float32)

        return batch, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)


class EpisodeBuffer:
    def __init__(self, with_hidden=False):
        self.memory = []
        self.Transition = Transition
        if with_hidden:
            self.Transition = DQRN_Transition
        self.position = 0

    def push(self, state, action, next_state, hidden, next_hidden, reward, done):
        self.memory.append(None)
        self.memory[self.position] = self.Transition(state, action, next_state, hidden, next_hidden, reward, done)
        self.position = self.position + 1

    def sample(self, lookup_step=None, idx=None):
        transitions = self.memory[idx:idx + lookup_step]
        batch = self.Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)


class EpisodeReplayMemory:
    def __init__(self, capacity, max_len=500, lookup_step=30):
        self.capacity = capacity
        self.max_len = max_len
        self.lookup_step = lookup_step
        self.memory = []
        self.position = 0

    def push(self, episode):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sampled_buffer = []
        sampled_episodes = random.sample(self.memory, batch_size)
        min_step = self.max_len

        for episode in sampled_episodes:
            min_step = min(min_step, len(episode))

        for episode in sampled_episodes:
            if min_step > self.lookup_step:  # sample buffer with lookup_step size
                idx = np.random.randint(0, len(episode) - self.lookup_step + 1)
                sample = episode.sample(lookup_step=self.lookup_step, idx=idx)
                sampled_buffer.append(sample)
            else:
                idx = np.random.randint(0, len(episode) - min_step + 1)  # sample buffer with minstep size
                sample = episode.sample(lookup_step=min_step, idx=idx)
                sampled_buffer.append(sample)
        return sampled_buffer, len(sample.action)  # buffers, sequence_length

    def __len__(self):
        return len(self.memory)
