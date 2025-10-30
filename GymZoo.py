import gymnasium as gym

# 初始化环境
env = gym.make("Pendulum-v1", render_mode="human")

# 重置环境并获取第一次的观测
observation, info = env.reset(seed=42)

env_names = list(gym.envs.registry.keys())
for name in env_names:
    print(name)
    
print(f"Action space: {env.action_space}")  # 二维度离散空间 Discrete(2)
print(f"Sample action: {env.action_space.sample()}")  # 0 or 1

print(f"Observation space: {env.observation_space}")  # 四维度连续空间
# Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
print(f"Sample observation: {env.observation_space.sample()}") 

episode_over = False
while not episode_over:
    # 在这里插入你自己的策略
    action = env.action_space.sample()

    # 执行动作使环境运行一个时间步（状态转移）
    # 接收下一个观测，奖励，以及是否结束或者截断
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated  # 如果回合结束，跳出循环。多回合则注释掉。

    # 如果回合结束，重置环境以开始新的回合
    if terminated or truncated:
        observation, info = env.reset()

env.close()