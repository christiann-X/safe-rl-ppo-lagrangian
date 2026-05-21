"""
PPO-Lagrangian 安全强化学习算法实现
基于拉格朗日乘子法将安全约束融入PPO目标函数，平衡奖励最大化和安全约束满足

参考: OmniSafe PPOLag 官方实现
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# ============================================================
# 神经网络模块
# ============================================================

class Actor(nn.Module):
    """策略网络：输出高斯分布的均值和对数标准差"""
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, act_dim)
        # 可学习的对数标准差参数（初始std≈0.6，避免动作过度截断）
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.log_std.exp().clamp(min=1e-4).expand_as(mean)
        return mean, std

    def get_action(self, obs, deterministic=False):
        """根据观测采样动作"""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, obs, action):
        """评估给定动作的对数概率和熵"""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """价值网络：估计状态价值 V(s)"""
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.out(x)


# ============================================================
# 经验缓冲区
# ============================================================

class RolloutBuffer:
    """存储一个batch的轨迹数据，用于PPO更新"""
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.costs = []
        self.values = []
        self.cost_values = []
        self.dones = []

    def add(self, obs, action, log_prob, reward, cost, value, cost_value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.values.append(value)
        self.cost_values.append(cost_value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def get_tensors(self, device):
        """将所有数据转为tensor"""
        obs = torch.tensor(np.array(self.obs), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32, device=device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        costs = torch.tensor(self.costs, dtype=torch.float32, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        cost_values = torch.tensor(self.cost_values, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        return obs, actions, log_probs, rewards, costs, values, cost_values, dones

    def __len__(self):
        return len(self.obs)


# ============================================================
# PPO-Lagrangian 算法
# ============================================================

class PPOLagrangian:
    """
    PPO-Lagrangian算法

    核心思想：通过拉格朗日乘子λ将安全约束融入PPO目标函数
    - 最大化: E[reward - λ * cost]
    - 自适应调整λ: 当cost超过限制时增大λ加强约束，否则减小λ
    """
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit=1.0,
        hidden_dim=256,
        lr_actor=3e-4,
        lr_critic=1e-3,
        lr_cost_critic=1e-3,
        lr_lambda=1e-2,
        gamma=0.99,
        lam=0.95,           # GAE参数
        clip_eps=0.2,       # PPO裁剪参数
        target_kl=0.01,     # 目标KL散度（早停）
        cost_limit=0.0,     # 成本限制阈值
        lambda_init=1.0,    # 拉格朗日乘子初始值
        lambda_lr=1e-2,     # 拉格朗日乘子学习率
        lambda_max=10.0,    # 拉格朗日乘子上限，防止cost信号过度压制reward
        entropy_coef=0.01,  # 熵正则化系数
        device='cpu'
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.target_kl = target_kl
        self.cost_limit = cost_limit
        self.lambda_max = lambda_max
        self.entropy_coef = entropy_coef
        self.device = device
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # 初始化网络
        self.actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
        self.reward_critic = Critic(obs_dim, hidden_dim).to(device)
        self.cost_critic = Critic(obs_dim, hidden_dim).to(device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.reward_critic_optimizer = optim.Adam(self.reward_critic.parameters(), lr=lr_critic)
        self.cost_critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=lr_cost_critic)

        # 拉格朗日乘子 λ（可学习参数，始终 ≥ 0）
        # 使用SGD（非Adam）让λ增长更平稳可控
        self.lambda_param = torch.tensor(lambda_init, dtype=torch.float32,
                                         device=device, requires_grad=True)
        self.lambda_optimizer = optim.SGD([self.lambda_param], lr=lambda_lr)

        # 代价EMA（用于平滑λ更新，避免剧烈波动）
        self.cost_ema = None

        # 训练日志
        self.lambda_history = []

    def compute_gae(self, values, rewards, dones, final_value=0.0, gamma=None, lam=None):
        """计算广义优势估计 (Generalized Advantage Estimation)

        Args:
            values:  各步的V(s_t)
            rewards: 各步的r_t
            dones:   各步是否终止
            final_value: V(s_{T+1})，即缓冲区最后一步之后的状态价值
            gamma, lam: 可覆盖默认值
        """
        if gamma is None:
            gamma = self.gamma
        if lam is None:
            lam = self.lam

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = final_value  # 使用传入的下一状态价值
            else:
                next_value = values[t + 1]
            # 如果当前步已终止，则下一状态价值为0
            if dones[t]:
                next_value = 0.0
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, buffer, final_obs=None):
        """使用缓冲区数据进行一次PPO-Lagrangian更新

        Args:
            buffer:    经验缓冲区
            final_obs: 缓冲区最后一步之后的状态观测，用于计算最后一步的GAE
        """
        obs, actions, old_log_probs, rewards, costs, values, cost_values, dones = \
            buffer.get_tensors(self.device)

        # 计算最后状态的下一状态价值（用于GAE尾部处理）
        final_value = 0.0
        final_cost_value = 0.0
        if final_obs is not None:
            obs_tensor = torch.tensor(final_obs, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
            with torch.no_grad():
                final_value = self.reward_critic(obs_tensor).squeeze(-1).item()
                final_cost_value = self.cost_critic(obs_tensor).squeeze(-1).item()

        # 计算奖励和成本的优势函数
        reward_advantages, reward_returns = self.compute_gae(
            values, rewards, dones, final_value=final_value)
        cost_advantages, cost_returns = self.compute_gae(
            cost_values, costs, dones, final_value=final_cost_value)

        # 标准化优势函数
        reward_advantages = (reward_advantages - reward_advantages.mean()) / \
                            (reward_advantages.std() + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / \
                          (cost_advantages.std() + 1e-8)

        # ---------- 更新奖励Critic ----------
        reward_critic_loss = F.mse_loss(self.reward_critic(obs).squeeze(-1), reward_returns)
        self.reward_critic_optimizer.zero_grad()
        reward_critic_loss.backward()
        self.reward_critic_optimizer.step()

        # ---------- 更新成本Critic ----------
        cost_critic_loss = F.mse_loss(self.cost_critic(obs).squeeze(-1), cost_returns)
        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()

        # ---------- 更新策略网络（PPO clipped objective with Lagrangian）----------
        # 合并优势: A_combined = A_reward - λ * A_cost
        combined_advantages = reward_advantages - self.lambda_param.detach() * cost_advantages

        # 计算新旧策略概率比
        new_log_probs, entropy = self.actor.evaluate(obs, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # PPO裁剪目标
        surr1 = ratio * combined_advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * combined_advantages
        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------- 更新拉格朗日乘子λ ----------
        # λ的目标: 让平均成本 ≤ cost_limit
        # 使用EMA平滑代价以减少λ剧烈波动，加上L2正则化防止λ无限增长
        mean_cost = costs.mean()
        if self.cost_ema is None:
            self.cost_ema = mean_cost.detach()
        else:
            self.cost_ema = 0.9 * self.cost_ema + 0.1 * mean_cost.detach()
        # λ_loss = -λ * (EMA_cost - cost_limit) + 0.0005 * λ²  (弱L2 decay，允许λ更自由增长)
        lambda_decay = 0.0005 * self.lambda_param ** 2
        lambda_loss = -self.lambda_param * (self.cost_ema - self.cost_limit) + lambda_decay

        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()

        # 确保λ在[0, lambda_max]范围内
        with torch.no_grad():
            self.lambda_param.clamp_(min=0.0, max=self.lambda_max)

        # 记录λ历史
        self.lambda_history.append(self.lambda_param.item())

        # 计算KL散度用于早停判断
        with torch.no_grad():
            old_mean, old_std = self.actor(obs)
            old_dist = Normal(old_mean, old_std)
            new_mean, new_std = self.actor(obs)
            new_dist = Normal(new_mean, new_std)
            kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()

        return {
            'actor_loss': actor_loss.item(),
            'reward_critic_loss': reward_critic_loss.item(),
            'cost_critic_loss': cost_critic_loss.item(),
            'lambda': self.lambda_param.item(),
            'lambda_loss': lambda_loss.item(),
            'mean_cost': mean_cost.item(),
            'entropy': entropy.mean().item(),
            'kl': kl.item(),
            'ratio_mean': ratio.mean().item(),
        }

    def save(self, path):
        """保存模型参数"""
        torch.save({
            'hidden_dim': self.hidden_dim,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'actor': self.actor.state_dict(),
            'reward_critic': self.reward_critic.state_dict(),
            'cost_critic': self.cost_critic.state_dict(),
            'lambda': self.lambda_param.item(),
        }, path)

    def load(self, path):
        """加载模型参数"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.reward_critic.load_state_dict(checkpoint['reward_critic'])
        self.cost_critic.load_state_dict(checkpoint['cost_critic'])
        with torch.no_grad():
            self.lambda_param.fill_(checkpoint['lambda'])
