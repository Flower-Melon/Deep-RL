import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os

class Actor(nn.Module):
    """
    Actor 网络，用于输出动作的策略网络。
    """
    def  __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        
        # 对于连续动作空间，随机策略梯度输出均值和标准差
        self.u = nn.Linear(256, action_dim)
        self.sigma = nn.Linear(256, action_dim)

        self.max_action = max_action # 缩放动作范围
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        
        mu = self.max_action * T.tanh(self.u(a)) # 使用tanh函数将动作限制在[-1, 1]之间，然后乘以max_action进行缩放
        sigma = F.softplus(self.sigma(a)) + 1e-5 # 使用softplus函数确保标准差为正值
        
        return mu, sigma
    
class Critic(nn.Module):
    """
    Critic 网络，用于评估动作价值的网络。
    """
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1) # 输出单一的状态价值V值

    def forward(self, state):
        v = F.relu(self.l1(state))
        v = F.relu(self.l2(v))
        v = self.l3(v)
        return v
    
class PPO():
    """
    PPO算法的实现。
    """
    def __init__(self, state_dim, action_dim, max_action, device, 
                 clip_param=0.2, gamma=0.99, lr=3e-4, 
                 ppo_epochs=10, max_grad_norm=0.5, gae_lambda=0.95):
            
        """
        初始化PPO算法的各个组件。
        """ 
        # 初始化动作和状态价值网络
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim).to(device)
        
        # 其他参数
        self.max_action = max_action
        self.clip_param = clip_param # 截断参数
        self.gamma = gamma # 折扣因子
        self.ppo_epochs = ppo_epochs # PPO 训练轮次
        self.max_grad_norm = max_grad_norm # 梯度裁剪最大范数
        self.device = device # 设备
        self.gae_lambda = gae_lambda # GAE 参数
        
        # 优化器
        self.actor_optimizer = T.optim.Adam(self.actor.parameters(), lr=3e-5)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(), lr=lr)
        
        # 经验池
        self.memory = []
        
    # select_action直接与环境交互，因此需要保证state和action都是numpy格式
    def select_action(self, state):
        """
        根据当前状态选择动作
        """
        state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.device)
        
        with T.no_grad():
            mu, sigma = self.actor(state)
            dist = T.distributions.Normal(mu, sigma) 
            action = dist.sample()
            action = T.clamp(action, -self.max_action, self.max_action)
            action_log = dist.log_prob(action).sum(dim=-1)                 
        return action.cpu().squeeze(0).numpy(), action_log.item()

    def store_transition(self, s, a, r, s_, log_prob, done):
        """
        存储一次经验（transition）
        """
        self.memory.append((s, a, r, s_, log_prob, done))

    def clear_memory(self):
        """
        清空经验池，PPO是on-policy算法，每次更新后都需要清空
        """
        self.memory = []
        
    def update(self, batch_size):
        """
        更新Actor和Critic网络的参数。
        """
        if len(self.memory) == 0:
            return

        # 从 memory 中提取数据并转换为张量
        states = T.tensor(np.array([t[0] for t in self.memory]), dtype=T.float).to(self.device)
        actions = T.tensor(np.array([t[1] for t in self.memory]), dtype=T.float).to(self.device)
        rewards = T.tensor([t[2] for t in self.memory], dtype=T.float).to(self.device).view(-1, 1)
        next_states = T.tensor(np.array([t[3] for t in self.memory]), dtype=T.float).to(self.device)
        old_log_probs = T.tensor([t[4] for t in self.memory], dtype=T.float).to(self.device).view(-1, 1)
        dones = T.tensor([t[5] for t in self.memory], dtype=T.float).to(self.device).view(-1, 1)
        
        # 归一化奖励，提升训练稳定性
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-8 # 防止除以零
        rewards = (rewards - reward_mean) / reward_std

        memory_size = len(self.memory)

        # 计算所有的 V-target 和 Advantage，注意这里全部分离梯度
        with T.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            
            # 计算 V-target (Critic 的目标)
            # v_targets = rewards + self.gamma * next_values * (1.0 - dones)
            
            # 计算优势 (Actor 的目标)
            # advantages = v_targets - values
            
            # GAE 计算
            advantages = T.zeros_like(rewards).to(self.device)
            last_gae_lam = 0

            for t in reversed(range(memory_size)):
                # dones[t] 决定了 next_values[t] 是否为 0
                next_non_terminal = 1.0 - dones[t]

                # GAE 的 delta (1-step TD 误差)
                delta = rewards[t] + self.gamma * next_values[t] * next_non_terminal - values[t]

                # GAE 优势累加
                advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

            # v_targets (Critic的目标) = GAE优势 + V值
            v_targets = advantages + values
            
            # 归一化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # PPO 训练循环（更新ppo_epochs轮次）
        for _ in range(self.ppo_epochs):
            
            # 随机打乱数据索引
            indices = T.randperm(memory_size)
            
            # 按照 batch_size 划分 mini-batch，进行多次更新
            for start in range(0, memory_size, batch_size):
                end = start + batch_size
                if end > memory_size:
                    break  # 如果最后一个 batch 不足 batch_size，则跳过

                batch_indices = indices[start:end]  # 随机采样的 mini-batch 索引

                # 从经验池中提取 mini-batch 数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_v_targets = v_targets[batch_indices]
                
                # 计算 Actor 损失 (PPO-Clip)
                mu, sigma = self.actor(batch_states)
                dist = T.distributions.Normal(mu, sigma)
                # new_log_probs 形状为 (batch_size, 1)，与 batch_old_log_probs 形状匹配
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1).view(-1, 1)
                
                ratios = T.exp(new_log_probs - batch_old_log_probs)
                
                # 计算 PPO-Clip surrogate loss
                surr1 = ratios * batch_advantages  # 未截断的 surrogate
                surr2 = T.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages  # 截断后的 surrogate

                # 增加熵正则化，鼓励探索
                ent_bonus = dist.entropy().mean()
                ent_coef = 0.01 # 熵正则化系数

                actor_loss = -T.min(surr1, surr2).mean() - ent_coef * ent_bonus  # 取最小值并取负号作为损失
                
                # 计算 Critic 损失
                current_values = self.critic(batch_states)
                critic_loss = F.smooth_l1_loss(current_values, batch_v_targets)
                
                # 更新 Actor 和 Critic 网络
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

        # 清空经验池 (On-policy)
        self.clear_memory()
        
    def save(self, directory):
        """
        用于保存模型，可以使用状态字典来保存和加载模型的参数，以便在训练后恢复模型的状态
        state_dict() 方法返回一个字典，其中键是参数的名称（通常是层的名称），值是对应的参数张量（torch.Tensor）
        """
        os.makedirs(directory, exist_ok=True)  # 自动创建目录
        T.save(self.actor.state_dict(), directory + 'actor.pth')
        T.save(self.critic.state_dict(), directory + 'critic.pth')

    def load(self, directory):
        """
        加载已保存的模型参数
        """
        self.actor.load_state_dict(T.load(directory + 'actor.pth'))
        self.critic.load_state_dict(T.load(directory + 'critic.pth'))
        
        print("====================================")
        print("model has been loaded...")
        print("====================================")

if __name__ == "__main__":
    # 测试 PPO 类的功能
    state_dim = 4
    action_dim = 2
    max_action = 1.0
    device = T.device("cpu")

    ppo = PPO(state_dim, action_dim, max_action, device)

    # 构造一个假状态
    state = np.random.randn(state_dim)
    action, log_prob = ppo.select_action(state)
    print("Selected action:", action)
    print("Log probability:", log_prob)

    # 存储一个假 transition
    next_state = np.random.randn(state_dim)
    reward = 1.0
    done = False
    ppo.store_transition(state, action, reward, next_state, log_prob, done)

    # 再存储几个假 transition
    for _ in range(5):
        s = np.random.randn(state_dim)
        a, lp = ppo.select_action(s)
        r = np.random.randn()
        s_ = np.random.randn(state_dim)
        d = np.random.choice([False, True])
        ppo.store_transition(s, a, r, s_, lp, d)

    # 更新网络
    ppo.update(batch_size=2)