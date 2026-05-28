import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PPO import PPO
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', default='train', type=str) # mode 为 'train' 或 'test'
    parser.add_argument('--env_name', default='BipedalWalker-v3', type=str) # 环境名称
    # parser.add_argument('--env_name', default='Pendulum-v1', type=str) # 环境名称
    parser.add_argument('--seed', default=42, type=int) # 随机种子
    
    parser.add_argument('--iteration', default=2000, type=int) # 训练迭代的(回合)次数
    parser.add_argument('--learning_rate', default=3e-4, type=float) # 学习率
    parser.add_argument('--batch_size', default=256, type=int) # PPO 更新时的 mini-batch 大小
    parser.add_argument('--max_step', default=1600, type=int) # 一个回合的最大步数
    parser.add_argument('--save_gif', default=False, type=bool) # 测试模式下是否保存 GIF
    parser.add_argument('--gif_path', default='./results/', type=str) # GIF 保存路径
    
    # PPO 特有参数
    parser.add_argument('--gamma', default=0.99, type=float) # 折扣因子
    parser.add_argument('--clip_param', default=0.2, type=float) # PPO 裁剪参数
    parser.add_argument('--ppo_epochs', default=5, type=int) # 每次 update 时训练的轮次
    parser.add_argument('--max_grad_norm', default=0.5, type=float) # 梯度裁剪最大范数
    parser.add_argument('--gae_lambda', default=0.95, type=float) # GAE 参数
    
    parser.add_argument('--load_model', default=False, type=bool) # 是否加载模型
    parser.add_argument('--model_path', default=None, type=str) # 模型路径
    parser.add_argument('--saveStep', default=100, type=int) # 每隔多少回合保存模型
    # 在 parse_args() 中
    parser.add_argument('--buffer_size', default=1024, type=int) # 收集多少步数据后再更新
    
    return parser.parse_args()

def main():
    args = parse_args()
    if args.model_path is None:
        args.model_path = './models/' + args.env_name + '/'
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 设置随机种子
    np.random.seed(args.seed)
    T.manual_seed(args.seed)
    
    # 创建环境
    render_mode = "rgb_array" if args.mode == 'test' else None
    env = gym.make(args.env_name, render_mode=render_mode)
    
    # 获取状态和动作的维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # 创建智能体
    agent = PPO(state_dim, action_dim, max_action, device,
                clip_param=args.clip_param,
                gamma=args.gamma,
                lr=args.learning_rate,
                ppo_epochs=args.ppo_epochs,
                max_grad_norm=args.max_grad_norm,
                gae_lambda=args.gae_lambda)
    
    # 训练模型
    if args.mode == 'train':
        print("====================================")
        print("Training...")
        print("====================================")
        
        # 加载模型
        if args.load_model:
            try:
                agent.load(args.model_path)
                print(f"Loaded model from {args.model_path}")
            except FileNotFoundError:
                print(f"No model found at {args.model_path}, starting from scratch.")
        
        episode_rewards = []
        actor_losses = []
        critic_losses = []

        for episode in range(args.iteration):

            # 重置环境
            state, _ = env.reset() # 为每个回合设置不同种子
            episode_reward = 0

            for step in range(args.max_step):

                # PPO: select_action 返回动作和对数概率
                action, action_log_prob = agent.select_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)

                # 终止标志
                done = terminated or truncated

                # PPO: 存储完整的 transition，包括 log_prob
                agent.store_transition(state, action, reward, next_state, action_log_prob, done)

                # 奖励累加
                episode_reward += reward
                # 转移到下一个状态
                state = next_state

                if len(agent.memory) >= args.buffer_size:
                    a_loss, c_loss = agent.update(args.batch_size)
                    if a_loss is not None:
                        actor_losses.append(a_loss)
                        critic_losses.append(c_loss)

                # 跳出循环
                if done:
                    break

            episode_rewards.append(episode_reward)    
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode: {episode + 1}, Steps: {step + 1}, Reward: {episode_reward:.2f}, Avg(10): {avg_reward:.2f}")
                
            if (episode + 1) % args.saveStep == 0:
                agent.save(args.model_path)
                print(f"Model saved at episode {episode + 1}")
                
        os.makedirs(args.gif_path, exist_ok=True)

        plt.figure()
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'{args.env_name} - PPO Reward Curve')
        plt.savefig(os.path.join(args.gif_path, f'{args.env_name}_ppo_reward.png'))
        plt.close()

        if actor_losses:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(actor_losses)
            ax1.set_xlabel('Update')
            ax1.set_ylabel('Actor Loss')
            ax1.set_title('Actor Loss')
            ax2.plot(critic_losses)
            ax2.set_xlabel('Update')
            ax2.set_ylabel('Critic Loss')
            ax2.set_title('Critic Loss')
            fig.suptitle(f'{args.env_name} - PPO Loss Curves')
            plt.tight_layout()
            plt.savefig(os.path.join(args.gif_path, f'{args.env_name}_ppo_loss.png'))
            plt.close()

        print(f"Plots saved to {args.gif_path}")

    elif args.mode == 'test':
        print("====================================")
        print("Testing...")
        print("====================================")
        
        # 加载模型
        try:
            agent.load(args.model_path)
            print(f"Loaded model from {args.model_path}")
        except FileNotFoundError:
            print(f"Error: No model found at {args.model_path}. Cannot run test.")
            env.close()
            return

        episode_rewards = []
        best_reward = -float('inf')
        best_frames = None

        for episode in range(10): # 测试10个回合
            state, _ = env.reset()
            episode_reward = 0
            frames = [] if args.save_gif else None

            for step in range(args.max_step):

                if frames is not None:
                    frame = env.render()
                    frames.append(Image.fromarray(frame))

                # 测试时，我们使用确定性的均值(mu)动作，而不是随机采样
                state_tensor = T.tensor(state, dtype=T.float).unsqueeze(0).to(device)
                with T.no_grad():
                    mu, _ = agent.actor(state_tensor) # 只取均值
                action = mu.cpu().squeeze(0).numpy()
                action = np.clip(action, -max_action, max_action) # 裁剪

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state

                if done:
                    break

            episode_rewards.append(episode_reward)
            print(f"Episode: {episode + 1}, Steps: {step + 1}, Reward: {episode_reward:.2f}")

            if frames is not None and episode_reward > best_reward:
                best_reward = episode_reward
                best_frames = frames

        avg_reward = np.mean(episode_rewards)
        print(f"Average Reward over 10 episodes: {avg_reward:.2f}")

        os.makedirs(args.gif_path, exist_ok=True)
        summary_path = os.path.join(args.gif_path, f"{args.env_name}_ppo_test_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Env: {args.env_name}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Model: {args.model_path}\n\n")
            for i, r in enumerate(episode_rewards):
                f.write(f"Episode {i + 1}: {r:.2f}\n")
            f.write(f"\nAverage: {avg_reward:.2f}\n")
            f.write(f"Best:   {best_reward:.2f}\n")
        print(f"Summary saved to {summary_path}")

        if best_frames is not None:
            os.makedirs(args.gif_path, exist_ok=True)
            save_path = os.path.join(args.gif_path, f"{args.env_name}_ppo.gif")
            best_frames[0].save(save_path, save_all=True, append_images=best_frames[1:],
                               duration=40, loop=0)
            print(f"GIF saved to {save_path} (best episode reward: {best_reward:.2f})")
            
    env.close()
    
if __name__ == '__main__':
    main()
