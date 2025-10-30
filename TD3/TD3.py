import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import Replay_buffer
import os

class Actor(nn.Module):
    """
    Actor 网络，用于输出动作的策略网络。
    """
    def  __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action # 缩放动作范围
    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * T.tanh(self.l3(a)) # 使用tanh函数将动作限制在[-1, 1]之间，然后乘以max_action进行缩放
        return a

class Critic(nn.Module):
    """
    Critic 网络，用于评估动作价值的网络。
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 400) # 状态和动作作为输入，输入维度为(batch_size, state_dim + action_dim)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1) # 输出单一的Q值
    
    def forward(self, state, action):
        q = F.relu(self.l1(T.cat([state, action], 1))) # 拼接状态和动作，torch.cat可以接受列表或元组的输入
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

class TD3():
    """
    TD3算法的实现。
    """
    def __init__(self, state_dim, action_dim, max_action, device, 
                 policy_noise=0.2, noise_clip=0.5, gamma=0.99, policy_freq=2, tau=0.005):
        
        """
        初始化TD3算法的各个组件，包括Actor和Critic网络及其目标网络，优化器等。
        """  
        # 初始化动作和目标动作网络
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        
        # 初始化两个Critic网络和它们的目标网络
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1 = Critic(state_dim, action_dim).to(device)
        self.critic_target_2 = Critic(state_dim, action_dim).to(device)
        
        # 优化器，目标网络无需优化器
        self.actor_optimizer = T.optim.Adam(self.actor.parameters()) # .parameters()返回模型的所有参数
        self.critic_1_optimizer = T.optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = T.optim.Adam(self.critic_2.parameters())
        
        # 初始化目标网络，保证其初始参数与对应的网络相同
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())    
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        
        # 其他参数
        self.max_action = max_action # 动作范围
        self.num_training = 0 # 计数器，记录总的训练次数
        self.device = device # 设备
        self.policy_noise = policy_noise # 策略噪声
        self.noise_clip = noise_clip # 噪声裁剪范围
        self.gamma = gamma # 折扣因子
        self.policy_freq = policy_freq # 策略更新频率
        self.tau = tau # 软更新参数
        
        # 经验缓存区
        self.replay_buffer = Replay_buffer(state_dim, action_dim)
        
    def select_action(self, state):
        """
        根据当前状态选择动作
        """
        # unsqueeze(0) 后变成二维（(1, state_dim)，即变成一个“batch”，方便神经网络处理
        state = T.unsqueeze(T.tensor(state, dtype=T.float), 0).to(self.device)
        # 选择动作并转换为numpy数组，flatten()将多维数组展平为一维,方便与环境交互
        return self.actor(state).data.cpu().numpy().flatten()
    
    def update(self, batch_size):
        """
        更新Actor和Critic网络的参数。
        """
        # 从经验回放缓冲区采样
        batch_s, batch_a, batch_r, batch_s_, batch_dw = self.replay_buffer.sample(batch_size)
        batch_s = batch_s.to(self.device)
        batch_a = batch_a.to(self.device)
        batch_r = batch_r.to(self.device)
        batch_s_ = batch_s_.to(self.device)
        batch_dw = batch_dw.to(self.device)
        
        # 计算TD目标时应该剥离计算图，以避免梯度传播到目标网络
        with T.no_grad():
            
            # trick1:在动作中添加噪声
            noise = (T.randn_like(batch_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) # 噪声
            a_ = (self.actor_target(batch_s_) + noise).clamp(-self.max_action, self.max_action) # 目标动作
            
            # trick2:双重Critic网络，取最小值
            target_q1 = self.critic_target_1(batch_s_, a_)
            target_q2 = self.critic_target_2(batch_s_, a_)
            target_q = T.min(target_q1, target_q2)
            # 计算目标Q值，0.99 是折扣因子 gamma，表示未来奖励的衰减率， batch_dw 是 done 标志，表示当前状态是否为终止状态
            target_q = batch_r + (1 - batch_dw) * self.gamma * target_q
        
        # 更新价值网络
        current_q1 = self.critic_1(batch_s, batch_a)
        current_q2 = self.critic_2(batch_s, batch_a)
        critic_1_loss = F.mse_loss(current_q1, target_q) 
        critic_2_loss = F.mse_loss(current_q2, target_q) # 注意这里target_q不需要detach，因为它已经在with T.no_grad()中计算过了
        # 更新Critic1网络
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward() 
        self.critic_1_optimizer.step()
        # 更新Critic2网络
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # trick3:延迟更新策略网络和目标网络
        self.num_training += 1
        if self.num_training % self.policy_freq == 0:
            # 更新策略网络
            # 这里使用了Critic1网络来评估策略，.mean()是因为要把batch的Q值取平均
            actor_loss = -self.critic_1(batch_s, self.actor(batch_s)).mean() 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            # 使用zip函数同时遍历两个网络的参数
            for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                            
    def save(self, directory):
        """
        用于保存模型，可以使用状态字典来保存和加载模型的参数，以便在训练后恢复模型的状态
        state_dict() 方法返回一个字典，其中键是参数的名称（通常是层的名称），值是对应的参数张量（torch.Tensor）
        """
        os.makedirs(directory, exist_ok=True)  # 自动创建目录
        T.save(self.actor.state_dict(), directory + f'actor.pth')
        T.save(self.actor_target.state_dict(), directory + f'actor_target.pth')
        T.save(self.critic_1.state_dict(), directory + f'critic_1.pth')
        T.save(self.critic_target_1.state_dict(), directory + f'critic_target_1.pth')
        T.save(self.critic_2.state_dict(), directory + f'critic_2.pth')
        T.save(self.critic_target_2.state_dict(), directory + f'critic_target_2.pth')

    def load(self, directory):
        self.actor.load_state_dict(T.load(directory + f'actor.pth'))
        self.actor_target.load_state_dict(T.load(directory + f'actor_target.pth'))
        self.critic_1.load_state_dict(T.load(directory + f'critic_1.pth'))
        self.critic_target_1.load_state_dict(T.load(directory + f'critic_target_1.pth'))
        self.critic_2.load_state_dict(T.load(directory + f'critic_2.pth'))
        self.critic_target_2.load_state_dict(T.load(directory + f'critic_target_2.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


if __name__ == "__main__":
    # 简单测试 Actor 和 Critic 网络的输出维度
    state_dim = 3
    action_dim = 2
    max_action = 1.0
    device = T.device("cpu")

    td3 = TD3(state_dim, action_dim, max_action, device)

    state = np.random.randn(state_dim)
    action = td3.select_action(state)
    print("Selected action:", action)

    # 测试 Critic 网络
    state_batch = T.rand(4, state_dim)
    action_batch = T.randn(4, action_dim)
    q_value = td3.critic_1(state_batch, action_batch)
    print("Critic Q value shape:", q_value.shape)
    
    # 测试 Actor 网络
    action_value = td3.actor(state_batch)
    print("Actor action value shape:", action_value.shape)
    
    # 测试更新函数
    td3.replay_buffer.push(state, action, 1.0, state, 0.0) # 添加一些假数据
    td3.update(num_iteration=1, batch_size=1)
    print("Update function executed successfully.")