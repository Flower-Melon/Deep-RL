import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
import torch as T

from SAC import SAC
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str)
    # parser.add_argument('--env_name', default='Pendulum-v1', type=str) # 环境名称
    parser.add_argument('--env_name', default='BipedalWalker-v3', type=str)
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--iteration', default=500, type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--max_step', default=2000, type=int)
    parser.add_argument('--save_gif', default=False, type=bool) # 测试模式下是否保存 GIF
    parser.add_argument('--gif_path', default='./results/', type=str) # GIF 保存路径
    parser.add_argument('--start_steps', default=10000, type=int)
    parser.add_argument('--updates_per_step', default=1, type=int)

    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.2, type=float)
    parser.add_argument('--automatic_entropy_tuning', default=True, type=bool)

    parser.add_argument('--load_model', default=False, type=bool)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--saveStep', default=100, type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_path is None:
        args.model_path = './models/' + args.env_name + '/'
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    T.manual_seed(args.seed)

    render_mode = "rgb_array" if args.mode == 'test' else None
    env = gym.make(args.env_name, render_mode=render_mode)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SAC(state_dim, action_dim, max_action, device,
                gamma=args.gamma,
                tau=args.tau,
                lr=args.learning_rate,
                alpha=args.alpha,
                automatic_entropy_tuning=args.automatic_entropy_tuning)

    if args.mode == 'train':
        print("====================================")
        print("Training...")
        print("====================================")

        if args.load_model:
            try:
                agent.load(args.model_path)
                print(f"Loaded model from {args.model_path}")
            except FileNotFoundError:
                print(f"No model found at {args.model_path}, starting from scratch.")

        episode_rewards = []
        total_steps = 0
        c1_losses = []
        c2_losses = []
        a_losses = []
        alpha_losses = []

        for episode in range(args.iteration):
            state, _ = env.reset(seed=args.seed + episode)
            episode_reward = 0

            for step in range(args.max_step):
                if total_steps < args.start_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.replay_buffer.push(state, action, reward, next_state, float(done))

                state = next_state
                episode_reward += reward
                total_steps += 1

                if len(agent.replay_buffer) > args.batch_size:
                    for _ in range(args.updates_per_step):
                        c1, c2, a, al = agent.update(args.batch_size)
                        c1_losses.append(c1)
                        c2_losses.append(c2)
                        a_losses.append(a)
                        if al is not None:
                            alpha_losses.append(al)

                if done:
                    break

            episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode: {episode + 1}, Steps: {step + 1}, Reward: {episode_reward:.2f}, Avg(10): {avg_reward:.2f}, Alpha: {agent.alpha:.4f}")

            if (episode + 1) % args.saveStep == 0:
                agent.save(args.model_path)
                print(f"Model saved at episode {episode + 1}")

        os.makedirs(args.gif_path, exist_ok=True)

        plt.figure()
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'{args.env_name} - SAC Reward Curve')
        plt.savefig(os.path.join(args.gif_path, f'{args.env_name}_sac_reward.png'))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(c1_losses, label='Critic 1')
        ax1.plot(c2_losses, label='Critic 2')
        ax1.set_xlabel('Update')
        ax1.set_ylabel('Loss')
        ax1.set_title('Critic Losses')
        ax1.legend()
        ax2.plot(a_losses, label='Actor')
        if alpha_losses:
            ax2.plot(alpha_losses, label='Alpha')
        ax2.set_xlabel('Update')
        ax2.set_ylabel('Loss')
        ax2.set_title('Actor / Alpha Loss')
        ax2.legend()
        fig.suptitle(f'{args.env_name} - SAC Loss Curves')
        plt.tight_layout()
        plt.savefig(os.path.join(args.gif_path, f'{args.env_name}_sac_loss.png'))
        plt.close()

        print(f"Plots saved to {args.gif_path}")

    elif args.mode == 'test':
        print("====================================")
        print("Testing...")
        print("====================================")

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

        for episode in range(10):
            state, _ = env.reset(seed=args.seed + episode)
            episode_reward = 0
            frames = [] if args.save_gif else None

            for step in range(args.max_step):

                if frames is not None:
                    frame = env.render()
                    frames.append(Image.fromarray(frame))

                action = agent.select_action(state, evaluate=True)
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
        summary_path = os.path.join(args.gif_path, f"{args.env_name}_sac_test_summary.txt")
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
            save_path = os.path.join(args.gif_path, f"{args.env_name}_sac.gif")
            best_frames[0].save(save_path, save_all=True, append_images=best_frames[1:],
                               duration=40, loop=0)
            print(f"GIF saved to {save_path} (best episode reward: {best_reward:.2f})")

    env.close()


if __name__ == '__main__':
    main()
