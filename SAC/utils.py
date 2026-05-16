import numpy as np
import torch as T


class Replay_buffer():
    """
    Experience replay buffer used by off-policy algorithms such as SAC.
    """
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.count = 0
        self.size = 0

        self.s = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.a = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.r = np.zeros((self.max_size, 1), dtype=np.float32)
        self.s_ = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.dw = np.zeros((self.max_size, 1), dtype=np.float32)

    def push(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw

        self.count = (self.count + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)

        batch_s = T.tensor(self.s[index], dtype=T.float)
        batch_a = T.tensor(self.a[index], dtype=T.float)
        batch_r = T.tensor(self.r[index], dtype=T.float)
        batch_s_ = T.tensor(self.s_[index], dtype=T.float)
        batch_dw = T.tensor(self.dw[index], dtype=T.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw

    def __len__(self):
        return self.size
