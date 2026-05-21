"""
安全强化学习课程设计 —— PPO-Lagrangian 训练脚本
使用 Safety Gymnasium 官方库的 CarGoal 系列环境

环境场景说明:
  Goal0 - 入门难度：仅小车+目标，无危险区和障碍物
  Goal1 - 中等难度：增加紫色危险区和少量蓝色障碍物
  Goal2 - 高级难度：大量障碍物和多处危险区，路径复杂

算法: PPO-Lagrangian
  通过拉格朗日乘子法将安全约束融入PPO目标函数，
  在最大化奖励的同时最小化约束违反。

用法:
  python train.py                      # 默认训练 Goal0 (无渲染)
  python train.py --goal 1             # 训练 Goal1
  python train.py --goal 2 --render    # 训练 Goal2 并开启可视化
  python train.py --goal 0 --test      # 测试已保存的模型

参考:
  Safety Gymnasium: https://github.com/PKU-Alignment/safety-gymnasium
  OmniSafe PPOLag: https://omnisafe.readthedocs.io/en/stable/algorithms/on_policy/ppo_lag.html
"""

import argparse
import sys
import os
import time
import json
from datetime import datetime
import numpy as np
import torch

# 将当前目录加入路径，确保能导入ppo_lag模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import safety_gymnasium
from ppo_lag import PPOLagrangian, RolloutBuffer


# ============================================================
# 超参数配置
# ============================================================

# 共享基础参数（所有场景适用）
BASE_CONFIG = {
    'gamma': 0.995,             # 折扣因子（导航任务需要更长视距）
    'lam': 0.95,                # GAE lambda参数
    'clip_eps': 0.2,            # PPO裁剪范围 epsilon
    'target_kl': 0.02,          # 目标KL散度（略微放宽以加速学习）
    'cost_limit': 0.0,          # 每步成本限制（0=不允许违反安全约束）
    'steps_per_epoch': 2048,    # 每个epoch收集的环境步数
    'update_iters': 10,         # 每个epoch的PPO更新迭代次数
    'save_freq': 100,           # 每N个episode保存一次模型
    'log_freq': 10,             # 每N个episode打印一次日志
}

# 各场景独立参数（根据难度调整）
# Goal0 (入门): 40维输入，无障碍物，已基本收敛
# Goal1 (中等): 72维输入，少量障碍物+危险区，需更多探索
# Goal2 (高级): 72维输入，大量障碍物，需最大容量和最长训练
SCENE_CONFIG = {
    0: {
        'hidden_dim': 256,
        'lr_actor': 1e-4,
        'lr_critic': 3e-4,
        'lr_cost_critic': 3e-4,
        'lr_lambda': 5e-3,
        'lambda_init': 1.0,
        'lambda_max': 5.0,
        'entropy_coef': 0.01,
        'total_steps': 500_000,
    },
    1: {
        'hidden_dim': 512,
        'lr_actor': 3e-5,        # 较慢的actor学习率，防止策略塌缩
        'lr_critic': 1e-4,
        'lr_cost_critic': 1e-4,
        'lr_lambda': 2e-5,       # 极慢λ增长(比v2低250倍)，给智能体100+回合学习导航
        'lambda_init': 0.01,     # 极低初始值，先学会到达目标
        'lambda_max': 3.0,       # 适度上限，有约束但不压垮学习
        'entropy_coef': 0.04,    # 高探索率，维持策略多样性
        'total_steps': 2_500_000,
    },
    2: {
        'hidden_dim': 1024,
        'lr_actor': 3e-5,        # 更低学习率保证稳定收敛
        'lr_critic': 1e-4,
        'lr_cost_critic': 1e-4,
        'lr_lambda': 1e-2,
        'lambda_init': 0.05,     # λ从极小开始，避免早期过度惩罚破坏探索
        'lambda_max': 10.0,
        'entropy_coef': 0.03,    # 最大化探索以找到复杂迷宫中的安全路径
        'total_steps': 2_000_000,
    },
}

# Goal1 Phase 2 配置：加载Phase 1模型，增强λ约束以降低代价
PHASE2_CONFIG_GOAL1 = {
    'hidden_dim': 512,
    'lr_actor': 1e-5,         # 极慢微调，保护已有导航能力
    'lr_critic': 5e-5,
    'lr_cost_critic': 5e-5,
    'lr_lambda': 5e-5,        # Phase 1的2.5倍，缓慢但持续增长
    'lambda_init': 0.028,     # 从Phase 1最终值开始
    'lambda_max': 5.0,        # 足够高的上限，真正施加代价压力
    'entropy_coef': 0.02,     # 适度探索
    'total_steps': 1_500_000,
}

# Goal1 Phase 3 配置：加载Phase 2模型，加速λ增长进入有效约束区间
PHASE3_CONFIG_GOAL1 = {
    'hidden_dim': 512,
    'lr_actor': 5e-6,         # 极保守，保护导航能力不被λ冲垮
    'lr_critic': 5e-5,
    'lr_cost_critic': 5e-5,
    'lr_lambda': 2e-4,        # 4倍Phase 2，推动λ快速进入0.5~2.0有效区间
    'lambda_init': 0.05,      # 从Phase 2最终值开始
    'lambda_max': 2.0,        # 精确上限，防止策略完全崩塌
    'entropy_coef': 0.02,
    'total_steps': 1_000_000,
}

# Goal1 Phase 4 配置：加载Phase 3模型，大幅提高λ学习率
# 根因：per-step cost仅~0.06(6%步在危险区)，导致λ梯度极小
# 解决：lr_lambda提高到1e-2（Phase3的50倍），λ预期从0.11→~3
PHASE4_CONFIG_GOAL1 = {
    'hidden_dim': 512,
    'lr_actor': 5e-6,         # 保守，保护导航能力
    'lr_critic': 5e-5,
    'lr_cost_critic': 5e-5,
    'lr_lambda': 1e-2,        # Phase 3的50倍，补偿per-step cost数值极小的问题
    'lambda_init': 0.11,      # 从Phase 3最终值继续
    'lambda_max': 5.0,        # 允许λ增长到有效约束区间
    'entropy_coef': 0.02,
    'total_steps': 1_000_000,
}

# 可用的场景配置
SCENES = {
    0: {'id': 'SafetyCarGoal0-v0',
        'desc': 'Goal0 - 入门难度：仅小车+目标，无危险区和障碍物'},
    1: {'id': 'SafetyCarGoal1-v0',
        'desc': 'Goal1 - 中等难度：增加紫色危险区和少量蓝色障碍物'},
    2: {'id': 'SafetyCarGoal2-v0',
        'desc': 'Goal2 - 高级难度：大量障碍物和多处危险区，路径复杂'},
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PPO-Lagrangian 安全强化学习训练')
    parser.add_argument('--goal', type=int, default=0, choices=[0, 1, 2],
                        help='选择训练场景: 0=Goal0, 1=Goal1, 2=Goal2 (默认: 0)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='开启可视化渲染窗口（默认关闭以加速训练）')
    parser.add_argument('--test', action='store_true', default=False,
                        help='测试模式：加载已保存模型并可视化运行')
    parser.add_argument('--model', type=str, default=None,
                        help='指定模型路径（默认自动查找）')
    parser.add_argument('--resume', type=str, default=None, metavar='PATH',
                        help='从指定checkpoint继续训练（Phase 2等）')
    parser.add_argument('--phase3', action='store_true', default=False,
                        help='使用Phase 3配置（需配合--resume）')
    parser.add_argument('--phase4', action='store_true', default=False,
                        help='使用Phase 4配置（需配合--resume）')
    parser.add_argument('--steps', type=int, default=None,
                        help='总训练步数（默认: 1,000,000）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    return parser.parse_args()


def get_env_name(goal):
    """根据场景编号获取环境ID"""
    return SCENES[goal]['id']


def set_seed(seed):
    """设置所有随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    """主训练流程"""
    goal = args.goal
    env_name = get_env_name(goal)
    render_mode = "human" if args.render else None
    is_resume = args.resume is not None

    # Phase 2/3/4：使用专门的安全优化配置
    if is_resume and goal == 1:
        if args.phase4:
            cfg = PHASE4_CONFIG_GOAL1
            print("=" * 60)
            print("  Goal1 Phase 4 —— λ大幅加速，强力约束cost")
            print("=" * 60)
        elif args.phase3:
            cfg = PHASE3_CONFIG_GOAL1
            print("=" * 60)
            print("  Goal1 Phase 3 —— λ加速进入有效约束区间，精准降低代价")
            print("=" * 60)
        else:
            cfg = PHASE2_CONFIG_GOAL1
            print("=" * 60)
            print("  Goal1 Phase 2 —— 安全约束增强，降低代价")
            print("=" * 60)
    else:
        cfg = SCENE_CONFIG[goal]

    total_steps = args.steps if args.steps else cfg['total_steps']
    set_seed(args.seed)

    print("=" * 60)
    print("  安全强化学习课程设计 —— PPO-Lagrangian")
    print(f"  场景: {SCENES[goal]['desc']}")
    print(f"  环境ID: {env_name}")
    print(f"  隐藏层维度: {cfg['hidden_dim']}  |  总训练步数: {total_steps:,}")
    print(f"  渲染窗口: {'开启' if args.render else '关闭（加速训练）'}")
    print(f"  λ初始值: {cfg['lambda_init']}  |  λ上限: {cfg['lambda_max']}  |  熵系数: {cfg['entropy_coef']}")
    print("=" * 60)

    # ========== 1. 创建环境 ==========
    print("\n[1/4] 创建 Safety Gymnasium 环境...")
    env = safety_gymnasium.make(env_name, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    print(f"  观测空间: {obs_dim} 维向量 (激光雷达 + 自身状态)")
    print(f"  动作空间: {act_dim} 维连续动作 (线速度, 角速度)")
    print(f"  动作范围: [{env.action_space.low[0]:.0f}, {env.action_space.high[0]:.0f}]")

    # ========== 2. 初始化算法 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2/4] 初始化 PPO-Lagrangian 算法 (计算设备: {device})")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    agent = PPOLagrangian(
        obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit,
        hidden_dim=cfg['hidden_dim'],
        lr_actor=cfg['lr_actor'],
        lr_critic=cfg['lr_critic'],
        lr_cost_critic=cfg['lr_cost_critic'],
        lr_lambda=cfg['lr_lambda'],
        gamma=BASE_CONFIG['gamma'], lam=BASE_CONFIG['lam'],
        clip_eps=BASE_CONFIG['clip_eps'], target_kl=BASE_CONFIG['target_kl'],
        cost_limit=BASE_CONFIG['cost_limit'], lambda_init=cfg['lambda_init'],
        lambda_lr=cfg['lr_lambda'], lambda_max=cfg['lambda_max'],
        entropy_coef=cfg['entropy_coef'],
        device=device
    )

    # Phase 2：加载Phase 1模型权重
    if is_resume:
        resume_path = args.resume
        if not os.path.exists(resume_path):
            print(f"[错误] 模型文件不存在: {resume_path}")
            return
        agent.load(resume_path)
        print(f"  已加载Phase 1模型: {resume_path}")
        print(f"  λ从 {agent.lambda_param.item():.4f} 开始继续训练")

    # ========== 3. 创建模型保存目录 ==========
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)

    # ========== 4. 训练循环 ==========
    buffer = RolloutBuffer()
    obs, _ = env.reset()

    # 当前episode的累计统计
    ep_reward = 0.0
    ep_cost = 0.0
    ep_len = 0

    # 历史统计列表
    episode_rewards = []
    episode_costs = []
    episode_lengths = []

    # 日志控制：防止同一episode重复打印
    last_logged_episode = 0
    train_start_time = time.time()
    log_start_time = train_start_time
    log_start_step = 0

    print(f"\n[3/4] 开始训练...\n")
    print(f"{'Episode':>7s}  {'Steps':>9s}  {'AvgRew':>10s}  {'AvgCost':>10s}  "
          f"{'Lambda':>8s}  {'Updates':>7s}  {'FPS':>7s}")
    print("-" * 70)

    update_count = 0  # PPO更新计数器

    for step in range(1, total_steps + 1):
        # ---- 采样动作 ----
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = agent.actor.get_action(obs_tensor)
            value = agent.reward_critic(obs_tensor).squeeze(-1)
            cost_value = agent.cost_critic(obs_tensor).squeeze(-1)

        # ---- 执行动作 ----
        action_np = action.cpu().numpy().flatten()
        # safety_gymnasium step() 返回6个值: obs, reward, cost, terminated, truncated, info
        next_obs, reward, cost, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        cost = float(cost)  # 安全约束代价（紫色危险区/蓝色障碍物碰撞）

        # ---- 存入经验缓冲区 ----
        buffer.add(obs, action_np, log_prob.item(), reward, cost,
                   value.item(), cost_value.item(), done)

        ep_reward += reward
        ep_cost += cost
        ep_len += 1

        obs = next_obs

        # ---- 回合结束 ----
        if done:
            episode_rewards.append(ep_reward)
            episode_costs.append(ep_cost)
            episode_lengths.append(ep_len)

            obs, _ = env.reset()
            ep_reward = 0.0
            ep_cost = 0.0
            ep_len = 0

        # ---- PPO-Lagrangian 更新 ----
        if len(buffer) >= BASE_CONFIG['steps_per_epoch']:
            final_obs = obs  # 缓冲区最后一步之后的状态
            for i in range(BASE_CONFIG['update_iters']):
                update_info = agent.update(buffer, final_obs=final_obs)
                update_count += 1
                # KL散度早停：避免策略更新过大
                if update_info['kl'] > BASE_CONFIG['target_kl']:
                    break
            buffer.clear()

        # ---- 打印训练日志（仅在episode数变化时打印，避免刷屏） ----
        num_episodes = len(episode_rewards)
        if num_episodes > 0 and num_episodes % BASE_CONFIG['log_freq'] == 0 \
           and num_episodes != last_logged_episode:
            last_logged_episode = num_episodes

            elapsed = time.time() - log_start_time
            steps_elapsed = step - log_start_step
            fps = steps_elapsed / elapsed if elapsed > 0 else 0

            avg_reward = np.mean(episode_rewards[-BASE_CONFIG['log_freq']:])
            avg_cost = np.mean(episode_costs[-BASE_CONFIG['log_freq']:])

            print(f"{num_episodes:>7d}  {step:>9,d}  {avg_reward:>10.2f}  "
                  f"{avg_cost:>10.4f}  {agent.lambda_param.item():>8.3f}  "
                  f"{update_count:>7d}  {fps:>7.0f}")

            log_start_time = time.time()
            log_start_step = step

        # ---- 定期保存模型 ----
        if num_episodes > 0 and num_episodes % BASE_CONFIG['save_freq'] == 0 \
           and num_episodes > (getattr(train, '_last_save', 0) or 0):
            save_path = os.path.join(model_dir, f'ppo_lag_goal{goal}_ep{num_episodes}.pth')
            agent.save(save_path)
            print(f"  >>> 模型已保存: {save_path}")
            train._last_save = num_episodes

    # ========== 训练完成 ==========
    total_time = time.time() - train_start_time
    print("\n" + "=" * 60)
    print(f"  [4/4] 训练完成!")
    print(f"  总训练步数: {total_steps:,}")
    print(f"  总回合数: {len(episode_rewards)}")
    print(f"  PPO更新次数: {update_count}")
    print(f"  总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")

    # 统计最终结果
    if episode_rewards:
        recent_n = min(50, len(episode_rewards))
        print(f"  最近{recent_n}回合平均奖励: {np.mean(episode_rewards[-recent_n:]):.2f}")
        print(f"  最近{recent_n}回合平均代价: {np.mean(episode_costs[-recent_n:]):.6f}")
        print(f"  最终λ值: {agent.lambda_param.item():.4f}")

    # 保存最终模型（Phase 4用独立文件名，避免覆盖Phase 3）
    if args.phase4:
        final_path = os.path.join(model_dir, f'ppo_lag_goal{goal}_phase4_final.pth')
    else:
        final_path = os.path.join(model_dir, f'ppo_lag_goal{goal}_final.pth')
    agent.save(final_path)
    print(f"  最终模型: {final_path}")

    # 保存训练日志和评价
    save_training_log(
        goal=goal, cfg=cfg,
        episode_rewards=episode_rewards,
        episode_costs=episode_costs,
        episode_lengths=episode_lengths,
        lambda_history=agent.lambda_history,
        total_steps=total_steps,
        total_time=total_time,
        update_count=update_count,
        model_path=final_path,
    )

    print("=" * 60)

    env.close()


def test(args):
    """测试模式：加载已保存的模型并可视化运行"""
    goal = args.goal
    env_name = get_env_name(goal)

    # 确定模型路径
    model_path = args.model
    if model_path is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        model_path = os.path.join(model_dir, f'ppo_lag_goal{goal}_final.pth')

    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print(f"请先训练: python train.py --goal {goal}")
        return

    print("=" * 60)
    print(f"  测试模式 —— {SCENES[goal]['desc']}")
    print(f"  加载模型: {model_path}")
    print("=" * 60)

    # 创建环境（强制开启渲染）
    env = safety_gymnasium.make(env_name, render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 从checkpoint读取模型架构参数（兼容新旧格式）
    ckpt = torch.load(model_path, map_location='cpu')
    hidden_dim = ckpt.get('hidden_dim', None)
    if hidden_dim is None:
        # 旧格式checkpoint：从参数形状推断hidden_dim
        hidden_dim = ckpt['actor']['fc1.weight'].shape[0]
    agent = PPOLagrangian(
        obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit,
        hidden_dim=hidden_dim, device=device
    )
    agent.load(model_path)
    print(f"  模型λ值: {agent.lambda_param.item():.4f}\n")

    # 运行测试
    num_test = 10
    total_reward = 0.0
    total_cost = 0.0

    for ep in range(1, num_test + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_cost = 0.0
        ep_len = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _ = agent.actor.get_action(obs_tensor, deterministic=True)
            action_np = action.cpu().numpy().flatten()
            obs, reward, cost, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            ep_reward += reward
            ep_cost += float(cost)
            ep_len += 1

        total_reward += ep_reward
        total_cost += ep_cost
        print(f"  Episode {ep:>2}:  Reward={ep_reward:>8.2f}  "
              f"Cost={ep_cost:>8.4f}  Steps={ep_len}")

    print(f"\n  平均奖励: {total_reward/num_test:.2f}")
    print(f"  平均代价: {total_cost/num_test:.4f}")
    print("  测试完成!")

    env.close()


def save_training_log(goal, cfg, episode_rewards, episode_costs, episode_lengths,
                      lambda_history, total_steps, total_time, update_count, model_path):
    """保存训练配置、结果和评价到日志文件"""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(results_dir, f'training_goal{goal}_{timestamp}.json')

    # 计算分段统计
    n_eps = len(episode_rewards)
    if n_eps > 0:
        seg_size = max(1, n_eps // 5)
        segments = {
            'initial_20%':  {'reward': float(np.mean(episode_rewards[:seg_size])),
                            'cost': float(np.mean(episode_costs[:seg_size]))},
            'early_20-40%': {'reward': float(np.mean(episode_rewards[seg_size:2*seg_size])),
                            'cost': float(np.mean(episode_costs[seg_size:2*seg_size]))},
            'mid_40-60%':   {'reward': float(np.mean(episode_rewards[2*seg_size:3*seg_size])),
                            'cost': float(np.mean(episode_costs[2*seg_size:3*seg_size]))},
            'late_60-80%':  {'reward': float(np.mean(episode_rewards[3*seg_size:4*seg_size])),
                            'cost': float(np.mean(episode_costs[3*seg_size:4*seg_size]))},
            'final_20%':    {'reward': float(np.mean(episode_rewards[-seg_size:])),
                            'cost': float(np.mean(episode_costs[-seg_size:]))},
        }

        recent_n = min(50, n_eps)
        final_stats = {
            'total_episodes': n_eps,
            'total_steps': total_steps,
            'total_time_seconds': round(total_time, 1),
            'total_time_minutes': round(total_time / 60, 1),
            'ppo_updates': update_count,
            'final_avg_reward_last50': float(np.mean(episode_rewards[-recent_n:])),
            'final_avg_cost_last50': float(np.mean(episode_costs[-recent_n:])),
            'best_episode_reward': float(max(episode_rewards)),
            'best_episode_cost': float(min(episode_costs)),
            'final_lambda': float(lambda_history[-1]) if lambda_history else None,
            'lambda_trajectory': [float(x) for x in lambda_history[-100:]],  # 最后100步λ值
        }

        # 评价
        final_reward = final_stats['final_avg_reward_last50']
        final_cost = final_stats['final_avg_cost_last50']
        if final_cost < 0.01 and final_reward > 15:
            evaluation = "优：智能体已学会在满足安全约束的前提下高效导航到达目标"
        elif final_cost < 0.1 and final_reward > 10:
            evaluation = "良：智能体基本掌握安全导航能力，偶有轻微违规"
        elif final_cost < 0.5 or final_reward > 5:
            evaluation = "中：智能体在学习但尚未稳定，需要更多训练或参数调优"
        else:
            evaluation = "差：模型未收敛，建议调整超参数或增加训练步数"
    else:
        segments = {}
        final_stats = {'total_episodes': 0}
        evaluation = "无数据"

    log = {
        'timestamp': timestamp,
        'scene': SCENES[goal]['desc'],
        'env_id': SCENES[goal]['id'],
        'config': {
            'hidden_dim': cfg['hidden_dim'],
            'lr_actor': cfg['lr_actor'],
            'lr_critic': cfg['lr_critic'],
            'lr_cost_critic': cfg['lr_cost_critic'],
            'lr_lambda': cfg['lr_lambda'],
            'lambda_init': cfg['lambda_init'],
            'lambda_max': cfg['lambda_max'],
            'entropy_coef': cfg['entropy_coef'],
            'gamma': BASE_CONFIG['gamma'],
            'lam': BASE_CONFIG['lam'],
            'clip_eps': BASE_CONFIG['clip_eps'],
            'target_kl': BASE_CONFIG['target_kl'],
            'cost_limit': BASE_CONFIG['cost_limit'],
        },
        'segment_analysis': segments,
        'final_statistics': final_stats,
        'evaluation': evaluation,
        'model_path': model_path,
    }

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n  训练日志已保存: {log_path}")
    print(f"  评价: {evaluation}")

    return log_path

    env.close()


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        test(args)
    else:
        train(args)
