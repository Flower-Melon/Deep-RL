import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from utils import Replay_buffer


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6


class Actor(nn.Module):
    """
    Stochastic actor for continuous action spaces.
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        mu = self.mu(a)
        log_std = self.log_std(a).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = T.distributions.Normal(mu, std)

        z = dist.rsample()
        tanh_z = T.tanh(z)
        action = tanh_z * self.max_action

        # Correct log probability after tanh squashing and action scaling.
        log_prob = dist.log_prob(z) - T.log(self.max_action * (1 - tanh_z.pow(2)) + EPS)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        deterministic_action = T.tanh(mu) * self.max_action
        return action, log_prob, deterministic_action


class Critic(nn.Module):
    """
    Q network used by SAC critics.
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(T.cat([state, action], dim=1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class SAC():
    """
    Soft Actor-Critic implementation for continuous control.
    """
    def __init__(self, state_dim, action_dim, max_action, device,
                 gamma=0.99, tau=0.005, lr=3e-4, alpha=0.2,
                 automatic_entropy_tuning=True):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1 = Critic(state_dim, action_dim).to(device)
        self.critic_target_2 = Critic(state_dim, action_dim).to(device)

        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = T.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = T.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = T.optim.Adam(self.critic_2.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.automatic_entropy_tuning = automatic_entropy_tuning

        if self.automatic_entropy_tuning:
            self.target_entropy = -float(action_dim)
            self.log_alpha = T.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
            self.log_alpha = None
            self.alpha_optimizer = None

        self.replay_buffer = Replay_buffer(state_dim, action_dim)

    def select_action(self, state, evaluate=False):
        state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.device)

        with T.no_grad():
            action, _, deterministic_action = self.actor.sample(state)

        if evaluate:
            return deterministic_action.cpu().numpy().flatten()
        return action.cpu().numpy().flatten()

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        batch_s, batch_a, batch_r, batch_s_, batch_dw = self.replay_buffer.sample(batch_size)
        batch_s = batch_s.to(self.device)
        batch_a = batch_a.to(self.device)
        batch_r = batch_r.to(self.device)
        batch_s_ = batch_s_.to(self.device)
        batch_dw = batch_dw.to(self.device)

        with T.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(batch_s_)
            target_q1 = self.critic_target_1(batch_s_, next_action)
            target_q2 = self.critic_target_2(batch_s_, next_action)
            target_q = T.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = batch_r + (1 - batch_dw) * self.gamma * target_q

        current_q1 = self.critic_1(batch_s, batch_a)
        current_q2 = self.critic_2(batch_s, batch_a)
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_action, log_prob, _ = self.actor.sample(batch_s)
        q1_new = self.critic_1(batch_s, new_action)
        q2_new = self.critic_2(batch_s, new_action)
        min_q_new = T.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss_val = None
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            alpha_loss_val = alpha_loss.item()

        for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_1_loss.item(), critic_2_loss.item(), actor_loss.item(), alpha_loss_val

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        T.save(self.actor.state_dict(), directory + 'actor.pth')
        T.save(self.critic_1.state_dict(), directory + 'critic_1.pth')
        T.save(self.critic_2.state_dict(), directory + 'critic_2.pth')
        T.save(self.critic_target_1.state_dict(), directory + 'critic_target_1.pth')
        T.save(self.critic_target_2.state_dict(), directory + 'critic_target_2.pth')

        if self.automatic_entropy_tuning:
            T.save(self.log_alpha.detach().cpu(), directory + 'log_alpha.pth')

    def load(self, directory):
        self.actor.load_state_dict(T.load(directory + 'actor.pth', map_location=self.device))
        self.critic_1.load_state_dict(T.load(directory + 'critic_1.pth', map_location=self.device))
        self.critic_2.load_state_dict(T.load(directory + 'critic_2.pth', map_location=self.device))
        self.critic_target_1.load_state_dict(T.load(directory + 'critic_target_1.pth', map_location=self.device))
        self.critic_target_2.load_state_dict(T.load(directory + 'critic_target_2.pth', map_location=self.device))

        if self.automatic_entropy_tuning and os.path.exists(directory + 'log_alpha.pth'):
            log_alpha = T.load(directory + 'log_alpha.pth', map_location=self.device)
            self.log_alpha.data.copy_(log_alpha.to(self.device))
            self.alpha = self.log_alpha.exp().item()

        print("====================================")
        print("model has been loaded...")
        print("====================================")


if __name__ == "__main__":
    state_dim = 3
    action_dim = 2
    max_action = 1.0
    device = T.device("cpu")

    sac = SAC(state_dim, action_dim, max_action, device)

    state = np.random.randn(state_dim)
    action = sac.select_action(state)
    print("Selected action:", action)

    for _ in range(8):
        s = np.random.randn(state_dim)
        a = sac.select_action(s)
        r = np.random.randn()
        s_ = np.random.randn(state_dim)
        dw = np.random.choice([0.0, 1.0])
        sac.replay_buffer.push(s, a, r, s_, dw)

    sac.update(batch_size=4)
    print("Update function executed successfully.")
