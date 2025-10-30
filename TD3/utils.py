import numpy as np
import torch as T

class Replay_buffer():
    '''
    经验回放缓冲区，用于存储和采样智能体的经验数据。
    '''
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e6) # 最大存储容量
        self.count = 0 # 当前存储位置
        self.size = 0 # 当前存储的样本数量
        self.s = np.zeros((self.max_size, state_dim)) # 状态
        self.a = np.zeros((self.max_size, action_dim)) # 动作
        self.r = np.zeros((self.max_size, 1)) # 奖励
        self.s_ = np.zeros((self.max_size, state_dim)) # 下一个状态
        self.dw = np.zeros((self.max_size, 1)) 

    def push(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # 循环覆盖
        self.size = min(self.size + 1, self.max_size)  # 更新当前样本数量

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # 随机采样索引
        batch_s = T.tensor(self.s[index], dtype=T.float)
        batch_a = T.tensor(self.a[index], dtype=T.float)
        batch_r = T.tensor(self.r[index], dtype=T.float)
        batch_s_ = T.tensor(self.s_[index], dtype=T.float)
        batch_dw = T.tensor(self.dw[index], dtype=T.float)

        # 返回采样的批次数据
        return batch_s, batch_a, batch_r, batch_s_, batch_dw
    
    def __len__(self):
        return self.size


# 测试
if __name__ == "__main__":
    buffer = Replay_buffer(3, 1)
    for i in range(10):
        buffer.push(np.array([i,i+1,i+2]), np.array([i]), i, np.array([i+1,i+2,i+3]), 0)
    bs, ba, br, bs_, bdw = buffer.sample(4)
    print('状态：', bs)
    print('动作：', ba)
    print('奖励：', br)
    print('下一个状态：', bs_)
    print('是否终止：', bdw)