import argparse

import gymnasium as gym
import numpy as np
import torch as T

from SAC import SAC


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
    parser.add_argument('--start_steps', default=10000, type=int)
    parser.add_argument('--updates_per_step', default=1, type=int)

    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.2, type=float)
    parser.add_argument('--automatic_entropy_tuning', default=True, type=bool)

    parser.add_argument('--load_model', default=False, type=bool)
    parser.add_argument('--model_path', default='./models/' + parser.get_default('env_name') + '/', type=str)
    parser.add_argument('--saveStep', default=100, type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    T.manual_seed(args.seed)

    env = gym.make(args.env_name, render_mode="human" if args.mode == 'test' else None)
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
                        agent.update(args.batch_size)

                if done:
                    break

            episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode: {episode + 1}, Steps: {step + 1}, Reward: {episode_reward:.2f}, Avg(10): {avg_reward:.2f}, Alpha: {agent.alpha:.4f}")

            if (episode + 1) % args.saveStep == 0:
                agent.save(args.model_path)
                print(f"Model saved at episode {episode + 1}")

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

        for episode in range(10):
            state, _ = env.reset(seed=args.seed + episode)
            episode_reward = 0

            for step in range(args.max_step):
                action = agent.select_action(state, evaluate=True)
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
