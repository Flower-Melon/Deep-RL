import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

import argparse
from TD3 import TD3

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', default='train', type=str) # mode 为 train 或 test
    # parser.add_argument('--env_name', default='Pendulum-v1', type=str) # 环境名称
    parser.add_argument('--env_name', default='BipedalWalker-v3', type=str) # 环境名称
    parser.add_argument('--seed', default=42, type=int) # 随机种子
    
    parser.add_argument('--iteration', default=500, type=int) # 训练迭代次数
    parser.add_argument('--learning_rate', default=3e-4, type=float) # 学习率
    parser.add_argument('--batch_size', default=256, type=int) # 批次
    parser.add_argument('--max_step', default=2000, type=float) # 一次epoch的最大步数
    
    parser.add_argument('--tau',  default=0.005, type=float) # 软更新参数
    parser.add_argument('--gamma', default=0.99, type=float) # 折扣因子
    parser.add_argument('--policy_noise', default=0.2, type=float) # 动作噪声
    parser.add_argument('--noise_clip', default=0.5, type=float) # 噪声裁剪
    parser.add_argument('--policy_freq', default=2, type=int) # 策略更新频率
    
    parser.add_argument('--load_model', default=False, type=bool) # 是否加载模型
    parser.add_argument('--model_path', default='./models/' + parser.get_default('env_name') + './', type=str) # 模型路径
    parser.add_argument('--saveStep', default=100, type=int) # 每隔多少回合保存模型
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建环境
    env = gym.make(args.env_name, render_mode="human" if args.mode == 'test' else None)
    
    # 获取状态和动作的维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # 创建智能体
    agent = TD3(state_dim, action_dim, max_action, device, 
                policy_noise=args.policy_noise, noise_clip=args.noise_clip, 
                gamma=args.gamma, policy_freq=args.policy_freq, tau=args.tau)
    
    # 训练模型
    if args.mode == 'train':
        print("====================================")
        print("Training...")
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        
        # 加载模型
        if args.load_model:
            agent.load(args.model_path)
            print(f"Loaded model from {args.model_path}")
        
        episode_rewards = []
        
        for episode in range(args.iteration):
            
            # 重置环境
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(args.max_step):
                
                # 选择动作，添加噪声以促进探索
                action = agent.select_action(state)
                action = (action + np.random.normal(0, max_action * 0.1, size=action_dim)).clip(-max_action, max_action)   
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # 终止标志
                done = terminated or truncated
                # 保存经验
                agent.replay_buffer.push(state, action, reward, next_state, float(done))
                # 奖励累加
                episode_reward += reward
                # 转移到下一个状态
                state = next_state
                
                # 只有当经验池中的样本数量大于批次大小时，才进行更新           
                if len(agent.replay_buffer) > args.batch_size:
                    agent.update(args.batch_size)
                    
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
                
    elif args.mode == 'test':
        print("====================================")
        print("Testing...")
        print("====================================")
        
        # 加载模型
        agent.load(args.model_path)
        print(f"Loaded model from {args.model_path}")
        
        for episode in range(10): # 测试10个回合
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(args.max_step):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            print(f"Episode: {episode + 1}, Steps: {step + 1}, Reward: {episode_reward:.2f}")
        
    env.close()
    
if __name__ == '__main__':
    main()   