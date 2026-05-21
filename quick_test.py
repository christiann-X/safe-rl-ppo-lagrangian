"""快速测试脚本：验证环境、导入和基本流程"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safety_gymnasium
import numpy as np
from ppo_lag import PPOLagrangian, RolloutBuffer

print("=" * 50)
print("1. 测试 Safety Gymnasium 环境")
print("=" * 50)

for goal in [0, 1, 2]:
    env_name = f'SafetyCarGoal{goal}-v0'
    env = safety_gymnasium.make(env_name)
    obs, _ = env.reset()
    print(f"\n{env_name}:")
    print(f"  观测: shape={obs.shape}, min={obs.min():.2f}, max={obs.max():.2f}")
    print(f"  动作空间: {env.action_space}")

    # 测试step (safety_gymnasium返回6个值)
    action = env.action_space.sample()
    next_obs, reward, cost, term, trunc, info = env.step(action)
    print(f"  动作采样: {action}")
    print(f"  奖励: {reward:.4f}, 代价: {cost:.4f}, 终止: {term}, 截断: {trunc}")
    print(f"  info keys: {list(info.keys())}")
    env.close()

print("\n" + "=" * 50)
print("2. 测试 PPO-Lagrangian 模块")
print("=" * 50)
import torch

# 测试网络
actor = PPOLagrangian(obs_dim=40, act_dim=2, device='cpu')
obs_tensor = torch.randn(1, 40)
action, log_prob = actor.actor.get_action(obs_tensor)
print(f"  动作采样: shape={action.shape}, 值={action.detach().numpy()[0]}")
print(f"  对数概率: {log_prob.item():.4f}")

value = actor.reward_critic(obs_tensor)
print(f"  价值估计: {value.item():.4f}")

# 测试buffer和update
buffer = RolloutBuffer()
obs_np = np.random.randn(40).astype(np.float32)
for _ in range(64):
    buffer.add(obs_np, np.random.randn(2).astype(np.float32),
               0.0, 1.0, 0.0, 0.5, 0.0, False)
update_info = actor.update(buffer, final_obs=obs_np)
print(f"\n  PPO更新结果:")
for k, v in update_info.items():
    print(f"    {k}: {v:.4f}")

print("\n全部测试通过!")
