"""
生成训练曲线图：Reward、Cost、Lambda 轨迹、分段分析、综合对比
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

# 训练结果文件及其标签
RESULT_FILES = {
    'Goal0':        'training_goal0_20260516_083233.json',
    'Goal1 v1(失败)': 'training_goal1_20260516_152357.json',
    'Goal1 Phase1': 'training_goal1_20260516_235159.json',
    'Goal1 Phase2': 'training_goal1_20260517_032923.json',
    'Goal1 Phase3': 'training_goal1_20260517_115422.json',
    'Goal1 Phase4': 'training_goal1_20260517_164545.json',
}

# 颜色方案
COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']
SEGMENT_COLORS = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

def load_results():
    data = {}
    for label, filename in RESULT_FILES.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data[label] = json.load(f)
        else:
            print(f"[警告] 文件不存在: {path}")
    return data


def plot_segment_comparison(data):
    """分段分析对比图：Reward 和 Cost"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    goal1_labels = [k for k in data if 'Goal1' in k]
    segments = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    segment_keys = ['initial_20%', 'early_20-40%', 'mid_40-60%', 'late_60-80%', 'final_20%']
    x = np.arange(len(segments))
    width = 0.13

    for i, label in enumerate(goal1_labels):
        d = data[label]
        rewards = [d['segment_analysis'][k]['reward'] for k in segment_keys]
        costs = [d['segment_analysis'][k]['cost'] for k in segment_keys]

        axes[0].bar(x + i * width, rewards, width, label=label, color=COLORS[i], alpha=0.85, edgecolor='white', linewidth=0.5)
        axes[1].bar(x + i * width, costs, width, label=label, color=COLORS[i], alpha=0.85, edgecolor='white', linewidth=0.5)

    for ax, title, ylabel in zip(axes,
                                  ['Goal1 各阶段 Reward 分段对比', 'Goal1 各阶段 Cost 分段对比'],
                                  ['Average Reward', 'Average Cost']):
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Progress', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(segments, fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'segment_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 分段对比图: {path}")


def plot_lambda_trajectories(data):
    """λ 轨迹图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：Goal1 Ph1-3（λ 增长缓慢的阶段）
    for i, label in enumerate(['Goal1 Phase1', 'Goal1 Phase2', 'Goal1 Phase3']):
        if label not in data:
            continue
        d = data[label]
        lambdas = d['final_statistics']['lambda_trajectory']
        axes[0].plot(range(len(lambdas)), lambdas, color=COLORS[i], linewidth=1.2,
                     label=f"{label} (lr={d['config']['lr_lambda']})", alpha=0.9)

    axes[0].set_title('Phase 1-3: λ 缓慢增长阶段', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('PPO Update', fontsize=12)
    axes[0].set_ylabel('Lambda (λ)', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # 右图：Goal1 Ph4（λ 快速增长）
    if 'Goal1 Phase4' in data:
        d = data['Goal1 Phase4']
        lambdas = d['final_statistics']['lambda_trajectory']
        axes[1].plot(range(len(lambdas)), lambdas, color=COLORS[4], linewidth=1.2,
                     label=f"Phase4 (lr=0.01)", alpha=0.9)

        # 标注关键点
        for idx, annotation in [(0, 'Start λ=0.11'), (-1, f'End λ={lambdas[-1]:.2f}')]:
            x_pos = idx if idx >= 0 else len(lambdas) - 1
            axes[1].annotate(annotation, xy=(x_pos, lambdas[x_pos]),
                           xytext=(x_pos + len(lambdas)*0.15, lambdas[x_pos] + 0.2),
                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                           fontsize=11, fontweight='bold', color='#e74c3c')

    axes[1].set_title('Phase 4: λ 大幅加速进入有效约束区间', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('PPO Update', fontsize=12)
    axes[1].set_ylabel('Lambda (λ)', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'lambda_trajectories.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] λ轨迹图: {path}")


def plot_phase_comparison(data):
    """综合对比图：最终 Reward/Cost/Lambda/训练时间"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    goal1_labels = [k for k in data if 'Goal1' in k]
    x_labels = [l.replace('Goal1 ', '') for l in goal1_labels]
    x = np.arange(len(x_labels))
    bar_colors = COLORS[:len(x_labels)]

    dlist = [data[l] for l in goal1_labels]

    # 1. 最终 Reward
    rewards = [d['final_statistics']['final_avg_reward_last50'] for d in dlist]
    bars = axes[0, 0].bar(x, rewards, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, rewards):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    axes[0, 0].set_title('Final Average Reward', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(x_labels, fontsize=10)
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)

    # 2. 最终 Cost
    costs = [d['final_statistics']['final_avg_cost_last50'] for d in dlist]
    bars = axes[0, 1].bar(x, costs, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, costs):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                       f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
    axes[0, 1].set_title('Final Average Cost', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(x_labels, fontsize=10)
    axes[0, 1].set_ylabel('Cost')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)

    # 3. 最终 Lambda
    lambdas = [d['final_statistics']['final_lambda'] for d in dlist]
    bars = axes[1, 0].bar(x, lambdas, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, lambdas):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                       f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    axes[1, 0].set_title('Final Lambda (λ)', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(x_labels, fontsize=10)
    axes[1, 0].set_ylabel('Lambda')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)

    # 4. 训练时间
    times = [d['final_statistics']['total_time_minutes'] for d in dlist]
    bars = axes[1, 1].bar(x, times, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, times):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                       f'{val:.0f}min', ha='center', fontsize=9, fontweight='bold')
    axes[1, 1].set_title('Training Time', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(x_labels, fontsize=10)
    axes[1, 1].set_ylabel('Minutes')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)

    plt.suptitle('Goal1 Training Comparison Across Phases', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'phase_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 阶段对比图: {path}")


def plot_reward_cost_scatter(data):
    """Reward vs Cost 散点图：各阶段最终表现 + 最优 episode"""
    fig, ax = plt.subplots(figsize=(10, 7))

    goal1_labels = [k for k in data if 'Goal1' in k]

    for i, label in enumerate(goal1_labels):
        d = data[label]
        final_r = d['final_statistics']['final_avg_reward_last50']
        final_c = d['final_statistics']['final_avg_cost_last50']
        best_r = d['final_statistics']['best_episode_reward']
        best_c = d['final_statistics']['best_episode_cost']

        # 最终平均
        ax.scatter(final_r, final_c, s=180, color=COLORS[i], edgecolors='white',
                  linewidth=1.5, zorder=5, alpha=0.9)
        ax.annotate(label.replace('Goal1 ', ''), (final_r, final_c),
                   textcoords="offset points", xytext=(10, 8),
                   fontsize=9, fontweight='bold', color=COLORS[i])

        # 最佳 episode（适当偏移避免重叠）
        offset_x, offset_y = (8, 5) if i % 2 == 0 else (-15, -10)
        ax.scatter(best_r, best_c, s=60, color=COLORS[i], edgecolors='white',
                  linewidth=1, zorder=5, alpha=0.5, marker='*')
        ax.annotate(f'Best ep', (best_r, best_c),
                   textcoords="offset points", xytext=(offset_x, offset_y),
                   fontsize=7, color=COLORS[i], alpha=0.7)

    # Goal0 参考
    if 'Goal0' in data:
        d0 = data['Goal0']
        ax.scatter(d0['final_statistics']['final_avg_reward_last50'],
                  d0['final_statistics']['final_avg_cost_last50'],
                  s=180, color=COLORS[0], edgecolors='white', linewidth=1.5,
                  zorder=5, alpha=0.9, marker='s')
        ax.annotate('Goal0', (d0['final_statistics']['final_avg_reward_last50'],
                              d0['final_statistics']['final_avg_cost_last50']),
                   textcoords="offset points", xytext=(10, -10),
                   fontsize=9, fontweight='bold', color=COLORS[0])

    ax.set_xlabel('Average Reward (higher is better)', fontsize=12)
    ax.set_ylabel('Average Cost (lower is better)', fontsize=12)
    ax.set_title('Reward-Cost Trade-off: Goal1 Training Evolution', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加理想区域标注
    ax.axvline(x=30, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=50, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.annotate('Ideal Zone\n(High Reward,\nLow Cost)', xy=(30.5, 30),
               fontsize=8, color='green', alpha=0.6)

    # 标注箭头：训练演化方向
    phase_order = ['Goal1 v1(失败)', 'Goal1 Phase1', 'Goal1 Phase2', 'Goal1 Phase3', 'Goal1 Phase4']
    ordered_labels = [l for l in phase_order if l in data]
    for i in range(len(ordered_labels) - 1):
        d1 = data[ordered_labels[i]]
        d2 = data[ordered_labels[i+1]]
        x1, y1 = d1['final_statistics']['final_avg_reward_last50'], d1['final_statistics']['final_avg_cost_last50']
        x2, y2 = d2['final_statistics']['final_avg_reward_last50'], d2['final_statistics']['final_avg_cost_last50']
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4,
                                  connectionstyle='arc3,rad=0.15', linewidth=0.8))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'reward_cost_scatter.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Reward-Cost散点图: {path}")


def plot_goal0_training(data):
    """Goal0 训练曲线（分段）"""
    if 'Goal0' not in data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    d = data['Goal0']
    segments = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    segment_keys = ['initial_20%', 'early_20-40%', 'mid_40-60%', 'late_60-80%', 'final_20%']

    rewards = [d['segment_analysis'][k]['reward'] for k in segment_keys]
    costs = [d['segment_analysis'][k]['cost'] for k in segment_keys]

    # Reward
    axes[0].fill_between(range(len(segments)), rewards, alpha=0.3, color=COLORS[0])
    axes[0].plot(range(len(segments)), rewards, 'o-', color=COLORS[0], linewidth=2, markersize=8)
    for i, (s, r) in enumerate(zip(segments, rewards)):
        axes[0].annotate(f'{r:.1f}', (i, r), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=10, fontweight='bold', color=COLORS[0])
    axes[0].set_xticks(range(len(segments)))
    axes[0].set_xticklabels(segments)
    axes[0].set_title('Goal0: Reward Growth', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Average Reward')
    axes[0].grid(alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Cost (always 0)
    axes[1].bar(segments, costs, color=COLORS[0], alpha=0.6, edgecolor='white')
    axes[1].set_title('Goal0: Cost (always 0 - no hazards)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Average Cost')
    axes[1].set_ylim(0, 5)
    axes[1].grid(alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'goal0_training.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Goal0训练图: {path}")


def plot_reward_cost_evolution(data):
    """Reward 和 Cost 随分段变化趋势（所有 Goal1 阶段）"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    goal1_labels = [k for k in data if 'Goal1' in k]
    segments = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    segment_keys = ['initial_20%', 'early_20-40%', 'mid_40-60%', 'late_60-80%', 'final_20%']

    for i, label in enumerate(goal1_labels):
        d = data[label]
        rewards = [d['segment_analysis'][k]['reward'] for k in segment_keys]
        costs = [d['segment_analysis'][k]['cost'] for k in segment_keys]
        x = np.arange(len(segments))

        axes[0].plot(x, rewards, 'o-', color=COLORS[i], linewidth=1.8, markersize=5,
                    label=label.replace('Goal1 ', ''), alpha=0.85)
        axes[1].plot(x, costs, 'o-', color=COLORS[i], linewidth=1.8, markersize=5,
                    label=label.replace('Goal1 ', ''), alpha=0.85)

    for ax, title, ylabel in zip(axes,
                                  ['Reward Evolution Across Training Segments',
                                   'Cost Evolution Across Training Segments'],
                                  ['Average Reward', 'Average Cost']):
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Training Progress', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(range(len(segments)))
        ax.set_xticklabels(segments)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'reward_cost_evolution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Reward-Cost演化图: {path}")


def plot_training_summary(data):
    """训练总览：表格式汇总图"""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')

    all_labels = ['Goal0'] + [k for k in data if 'Goal1' in k]
    rows = []
    for label in all_labels:
        if label not in data:
            continue
        d = data[label]
        cfg = d['config']
        stats = d['final_statistics']
        rows.append([
            label,
            f"{stats['final_avg_reward_last50']:.2f}",
            f"{stats['final_avg_cost_last50']:.1f}",
            f"{stats['final_lambda']:.3f}",
            f"{cfg['lr_lambda']}",
            f"{stats['total_steps']:,}",
            f"{stats['total_time_minutes']:.0f}",
            f"{stats['best_episode_reward']:.1f}",
            d['evaluation'].replace('：', ':'),
        ])

    col_labels = ['Training', 'Reward', 'Cost', 'λ', 'lr_λ', 'Steps', 'Time(min)', 'Best R', 'Evaluation']
    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')

    # 颜色映射
    eval_colors = {'优': '#2ecc71', '中': '#f39c12', '差': '#e74c3c'}
    for i, row in enumerate(rows):
        eval_text = row[-1]
        for key, color in eval_colors.items():
            if key in eval_text:
                table[(i + 1, len(col_labels) - 1)].set_facecolor(color)
                table[(i + 1, len(col_labels) - 1)].set_text_props(color='white', fontweight='bold')
                break

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)

    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    plt.title('Training Summary Table', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'training_summary_table.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 训练总览表: {path}")


def plot_demo_comparison(data):
    """Demo 测试对比图（手动数据）"""
    fig, ax = plt.subplots(figsize=(8, 6))

    models = ['Phase 3\n(λ=0.11)', 'Ep 600\n(λ=1.97)', 'Phase 4\n(λ=3.01)']
    rewards = [30.50, 26.69, 28.81]
    costs = [58.30, 69.60, 85.60]
    cost_ranges = [(17, 85), (33, 153), (0, 259)]

    x = np.arange(len(models))
    width = 0.3

    # Reward bars
    bars1 = ax.bar(x - width/2, rewards, width, label='Avg Reward', color='#3498db', alpha=0.85, edgecolor='white')
    for bar, val in zip(bars1, rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}', ha='center', fontsize=10, fontweight='bold', color='#3498db')

    # Cost bars on twin axis
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, costs, width, label='Avg Cost', color='#e74c3c', alpha=0.85, edgecolor='white')
    for bar, val in zip(bars2, costs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
               f'{val:.1f}', ha='center', fontsize=10, fontweight='bold', color='#e74c3c')

    # Cost range annotations
    for i, (lo, hi) in enumerate(cost_ranges):
        ax2.annotate(f'Range: [{lo}, {hi}]', (x[i] + width/2, costs[i]),
                    textcoords="offset points", xytext=(0, -20),
                    fontsize=7, ha='center', color='#e74c3c', alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel('Average Reward (↑)', color='#3498db', fontsize=12)
    ax2.set_ylabel('Average Cost (↓)', color='#e74c3c', fontsize=12)
    ax.set_title('Demo Test Comparison (10 episodes, deterministic)', fontsize=13, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'demo_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Demo对比图: {path}")


def plot_phase234_combined(data):
    """Goal1 Phase 2-4 结果曲线综合图"""
    labels = ['Goal1 Phase2', 'Goal1 Phase3', 'Goal1 Phase4']
    phase_colors = ['#9b59b6', '#f39c12', '#1abc9c']
    available = [(l, c) for l, c in zip(labels, phase_colors) if l in data]
    if len(available) < 2:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    segments = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    segment_keys = ['initial_20%', 'early_20-40%', 'mid_40-60%', 'late_60-80%', 'final_20%']
    x = np.arange(len(segments))

    # ---- 第一行：三个 Phase 各自的 Reward+Cost 双轴图 ----
    for idx, (label, color) in enumerate(available):
        d = data[label]
        rewards = [d['segment_analysis'][k]['reward'] for k in segment_keys]
        costs = [d['segment_analysis'][k]['cost'] for k in segment_keys]

        ax_r = axes[0, idx]
        ax_c = ax_r.twinx()

        # Reward 折线
        line_r, = ax_r.plot(x, rewards, 'o-', color=color, linewidth=2.2, markersize=8,
                           markerfacecolor='white', markeredgewidth=2, label='Reward')
        ax_r.fill_between(x, rewards, alpha=0.1, color=color)
        for i, (s, r) in enumerate(zip(segments, rewards)):
            ax_r.annotate(f'{r:.1f}', (i, r), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=9, fontweight='bold', color=color)

        # Cost 折线
        line_c, = ax_c.plot(x, costs, 's--', color='#e74c3c', linewidth=1.8, markersize=7,
                           markerfacecolor='white', markeredgewidth=1.5, label='Cost')
        for i, (s, c) in enumerate(zip(segments, costs)):
            ax_c.annotate(f'{c:.1f}', (i, c), textcoords="offset points",
                        xytext=(0, -18), ha='center', fontsize=8, color='#e74c3c')

        ax_r.set_xticks(x)
        ax_r.set_xticklabels(segments, fontsize=8)
        phase_name = label.replace('Goal1 ', '')
        ax_r.set_title(f'{phase_name}\n(λ: {d["final_statistics"]["final_lambda"]:.3f})',
                      fontsize=13, fontweight='bold', color=color)
        ax_r.set_ylabel('Reward', color=color, fontsize=10)
        ax_c.set_ylabel('Cost', color='#e74c3c', fontsize=10)
        ax_r.tick_params(axis='y', labelcolor=color)
        ax_c.tick_params(axis='y', labelcolor='#e74c3c')
        ax_r.grid(alpha=0.25)
        ax_r.spines['top'].set_visible(False)

        lines = [line_r, line_c]
        labels_leg = ['Reward', 'Cost']
        ax_r.legend(lines, labels_leg, fontsize=8, loc='upper left')

    # ---- 第二行左：Reward 同轴对比 ----
    for (label, color) in available:
        d = data[label]
        rewards = [d['segment_analysis'][k]['reward'] for k in segment_keys]
        axes[1, 0].plot(x, rewards, 'o-', color=color, linewidth=2, markersize=7,
                       markerfacecolor='white', markeredgewidth=1.8,
                       label=f"{label.replace('Goal1 ','')} (final λ={d['final_statistics']['final_lambda']:.3f})")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(segments)
    axes[1, 0].set_title('Reward Comparison: Phase 2 vs 3 vs 4', fontsize=13, fontweight='bold')
    axes[1, 0].set_ylabel('Average Reward', fontsize=11)
    axes[1, 0].legend(fontsize=8, loc='lower right')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)

    # ---- 第二行中：Cost 同轴对比 ----
    for (label, color) in available:
        d = data[label]
        costs = [d['segment_analysis'][k]['cost'] for k in segment_keys]
        axes[1, 1].plot(x, costs, 's-', color=color, linewidth=2, markersize=7,
                       markerfacecolor='white', markeredgewidth=1.8,
                       label=f"{label.replace('Goal1 ','')} (final λ={d['final_statistics']['final_lambda']:.3f})")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(segments)
    axes[1, 1].set_title('Cost Comparison: Phase 2 vs 3 vs 4', fontsize=13, fontweight='bold')
    axes[1, 1].set_ylabel('Average Cost', fontsize=11)
    axes[1, 1].legend(fontsize=8, loc='upper left')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)

    # ---- 第二行右：最终 Reward-Cost 散点 + λ 标注 ----
    for (label, color) in available:
        d = data[label]
        fr = d['final_statistics']['final_avg_reward_last50']
        fc = d['final_statistics']['final_avg_cost_last50']
        fl = d['final_statistics']['final_lambda']
        axes[1, 2].scatter(fr, fc, s=250, color=color, edgecolors='white',
                          linewidth=2, zorder=5, alpha=0.9)
        offset = 12 if label != 'Goal1 Phase3' else -18
        axes[1, 2].annotate(f"{label.replace('Goal1 ','')}\nλ={fl:.3f}\n(R={fr:.1f}, C={fc:.1f})",
                          (fr, fc), textcoords="offset points",
                          xytext=(offset, -8), fontsize=9, fontweight='bold', color=color)
    # 箭头：Phase2→Phase3→Phase4
    avail_labels = [l for l, _ in available]
    for i in range(len(avail_labels) - 1):
        d1, d2 = data[avail_labels[i]], data[avail_labels[i+1]]
        axes[1, 2].annotate('', xy=(d2['final_statistics']['final_avg_reward_last50'],
                                    d2['final_statistics']['final_avg_cost_last50']),
                          xytext=(d1['final_statistics']['final_avg_reward_last50'],
                                  d1['final_statistics']['final_avg_cost_last50']),
                          arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5,
                                        connectionstyle='arc3,rad=0.2', linewidth=1))
    axes[1, 2].set_xlabel('Final Reward', fontsize=11)
    axes[1, 2].set_ylabel('Final Cost', fontsize=11)
    axes[1, 2].set_title('Reward-Cost Trade-off Evolution', fontsize=13, fontweight='bold')
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].spines['top'].set_visible(False)
    axes[1, 2].spines['right'].set_visible(False)

    fig.suptitle('Goal1 Phase 2-4 训练结果综合分析', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'phase234_combined.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Phase2-4综合图: {path}")


def plot_phase1_standalone(data):
    """Goal1 Phase 1 结果曲线图（与 Phase 2-4 统一样式）"""
    label = 'Goal1 Phase1'
    if label not in data:
        return

    d = data[label]
    color = '#3498db'
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    segments = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    segment_keys = ['initial_20%', 'early_20-40%', 'mid_40-60%', 'late_60-80%', 'final_20%']
    x = np.arange(len(segments))
    rewards = [d['segment_analysis'][k]['reward'] for k in segment_keys]
    costs = [d['segment_analysis'][k]['cost'] for k in segment_keys]

    # ---- 左：Reward+Cost 双轴图 ----
    ax_r = axes[0]
    ax_c = ax_r.twinx()

    line_r, = ax_r.plot(x, rewards, 'o-', color=color, linewidth=2.5, markersize=10,
                       markerfacecolor='white', markeredgewidth=2, label='Reward')
    ax_r.fill_between(x, rewards, alpha=0.1, color=color)
    for i, (s, r) in enumerate(zip(segments, rewards)):
        ax_r.annotate(f'{r:.1f}', (i, r), textcoords="offset points",
                    xytext=(0, 14), ha='center', fontsize=10, fontweight='bold', color=color)

    line_c, = ax_c.plot(x, costs, 's--', color='#e74c3c', linewidth=2, markersize=8,
                       markerfacecolor='white', markeredgewidth=1.5, label='Cost')
    for i, (s, c) in enumerate(zip(segments, costs)):
        ax_c.annotate(f'{c:.1f}', (i, c), textcoords="offset points",
                    xytext=(0, -22), ha='center', fontsize=9, color='#e74c3c')

    ax_r.set_xticks(x)
    ax_r.set_xticklabels(segments, fontsize=9)
    ax_r.set_title('Phase 1: Reward + Cost\n(lr_lambda=2e-5, λ final=0.028)',
                  fontsize=13, fontweight='bold', color=color)
    ax_r.set_ylabel('Reward', color=color, fontsize=11)
    ax_c.set_ylabel('Cost', color='#e74c3c', fontsize=11)
    ax_r.tick_params(axis='y', labelcolor=color)
    ax_c.tick_params(axis='y', labelcolor='#e74c3c')
    ax_r.grid(alpha=0.25)
    ax_r.spines['top'].set_visible(False)
    ax_r.legend([line_r, line_c], ['Reward', 'Cost'], fontsize=9, loc='upper left')

    # ---- 中：λ 轨迹 ----
    lambdas = d['final_statistics']['lambda_trajectory']
    updates = np.arange(len(lambdas))
    axes[1].plot(updates, lambdas, color=color, linewidth=2, alpha=0.9)
    axes[1].fill_between(updates, lambdas, alpha=0.1, color=color)
    axes[1].annotate(f'Start: λ={lambdas[0]:.4f}', xy=(0, lambdas[0]),
                   xytext=(len(lambdas)*0.05, lambdas[0] + 0.0005),
                   fontsize=9, fontweight='bold', color=color,
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))
    axes[1].annotate(f'End: λ={lambdas[-1]:.4f}', xy=(len(lambdas)-1, lambdas[-1]),
                   xytext=(len(lambdas)*0.55, lambdas[-1] + 0.0005),
                   fontsize=9, fontweight='bold', color=color,
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))
    axes[1].set_title(f'λ Trajectory ({d["final_statistics"]["ppo_updates"]} PPO updates)',
                     fontsize=13, fontweight='bold')
    axes[1].set_xlabel('PPO Update', fontsize=11)
    axes[1].set_ylabel('Lambda (λ)', fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # ---- 右：训练信息卡片 ----
    axes[2].axis('off')
    info_items = [
        ('Training Info', '', 'header'),
        ('', '', 'spacer'),
        ('Total Episodes', f"{d['final_statistics']['total_episodes']:,}", 'data'),
        ('Total Steps', f"{d['final_statistics']['total_steps']:,}", 'data'),
        ('Training Time', f"{d['final_statistics']['total_time_minutes']:.0f} min ({d['final_statistics']['total_time_minutes']/60:.1f}h)", 'data'),
        ('PPO Updates', f"{d['final_statistics']['ppo_updates']:,}", 'data'),
        ('', '', 'spacer'),
        ('Results', '', 'header'),
        ('', '', 'spacer'),
        ('Final Reward', f"{d['final_statistics']['final_avg_reward_last50']:.2f}", 'data'),
        ('Final Cost', f"{d['final_statistics']['final_avg_cost_last50']:.1f}", 'data'),
        ('Best Reward', f"{d['final_statistics']['best_episode_reward']:.1f}", 'data'),
        ('Best Cost', f"{d['final_statistics']['best_episode_cost']:.1f}", 'data'),
        ('Final λ', f"{d['final_statistics']['final_lambda']:.6f}", 'data'),
        ('', '', 'spacer'),
        ('Evaluation', d['evaluation'].replace('：', ': '), 'eval'),
    ]

    y_pos = 0.95
    for item_name, item_value, item_type in info_items:
        if item_type == 'header':
            axes[2].text(0.05, y_pos, item_name, fontsize=13, fontweight='bold',
                       color='#2c3e50', transform=axes[2].transAxes)
            axes[2].plot([0.05, 0.95], [y_pos - 0.03, y_pos - 0.03],
                       linewidth=1.5, color=color, transform=axes[2].transAxes, clip_on=False)
            y_pos -= 0.07
        elif item_type == 'spacer':
            y_pos -= 0.02
        elif item_type == 'data':
            axes[2].text(0.10, y_pos, f'{item_name}:', fontsize=10, fontweight='bold',
                       color='#555', transform=axes[2].transAxes)
            axes[2].text(0.55, y_pos, item_value, fontsize=10,
                       color='#333', transform=axes[2].transAxes)
            y_pos -= 0.06
        elif item_type == 'eval':
            eval_color = '#2ecc71' if '优' in item_value else ('#f39c12' if '中' in item_value else '#e74c3c')
            axes[2].text(0.10, y_pos, f'{item_name}:', fontsize=11, fontweight='bold',
                       color='#555', transform=axes[2].transAxes)
            axes[2].text(0.55, y_pos, item_value, fontsize=11, fontweight='bold',
                       color=eval_color, transform=axes[2].transAxes)

    fig.suptitle('Goal1 Phase 1 训练结果', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'phase1_standalone.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Phase1独立图: {path}")


def plot_phase4_detailed(data):
    """Phase 4 详细训练曲线（逐10-episode数据，标注最佳平衡点）"""
    if 'Goal1 Phase4' not in data:
        return

    # 从训练日志提取的每10-episode数据: (ep, reward, cost, lambda)
    log_data = [
        (10, 29.67, 72.9, 0.143), (20, 32.20, 37.3, 0.168), (30, 31.53, 59.6, 0.194),
        (40, 30.66, 47.5, 0.214), (50, 31.83, 70.7, 0.253), (60, 29.98, 60.9, 0.285),
        (70, 31.35, 79.0, 0.319), (80, 30.63, 74.8, 0.357), (90, 32.10, 86.6, 0.392),
        (100, 30.68, 90.6, 0.437), (110, 29.21, 65.9, 0.476), (120, 31.71, 69.0, 0.509),
        (130, 30.35, 75.4, 0.543), (140, 31.86, 53.1, 0.569), (150, 30.48, 57.9, 0.599),
        (160, 30.61, 83.6, 0.637), (170, 30.97, 74.8, 0.676), (180, 30.02, 74.8, 0.703),
        (190, 29.15, 94.9, 0.744), (200, 31.44, 53.8, 0.779),
        (210, 29.94, 64.1, 0.809), (220, 30.44, 81.0, 0.845), (230, 30.03, 70.3, 0.884),
        (240, 30.87, 41.5, 0.911), (250, 30.24, 74.9, 0.942), (260, 29.83, 56.5, 0.968),
        (270, 29.46, 54.3, 0.993), (280, 30.06, 55.3, 1.022), (290, 31.02, 65.4, 1.052),
        (300, 28.76, 38.8, 1.076), (310, 30.16, 54.8, 1.099), (320, 30.64, 68.6, 1.132),
        (330, 30.38, 25.7, 1.148), (340, 30.47, 54.2, 1.172), (350, 30.53, 84.3, 1.208),
        (360, 29.48, 63.9, 1.237), (370, 29.11, 43.4, 1.258), (380, 29.33, 29.2, 1.280),
        (390, 30.44, 47.0, 1.297), (400, 29.13, 69.9, 1.333),
        (410, 30.80, 61.5, 1.358), (420, 30.27, 88.2, 1.402), (430, 30.32, 64.5, 1.427),
        (440, 30.29, 44.6, 1.456), (450, 30.34, 50.0, 1.477), (460, 29.27, 84.0, 1.511),
        (470, 30.84, 58.9, 1.545), (480, 32.01, 41.2, 1.567), (490, 29.75, 85.6, 1.603),
        (500, 30.50, 48.1, 1.633), (510, 30.71, 71.2, 1.662), (520, 31.28, 70.0, 1.688),
        (530, 30.38, 57.3, 1.716), (540, 29.46, 63.4, 1.752), (550, 29.97, 62.7, 1.783),
        (560, 29.87, 100.2, 1.827), (570, 28.88, 98.2, 1.876), (580, 30.44, 74.9, 1.913),
        (590, 30.43, 77.0, 1.947), (600, 29.21, 48.2, 1.971),
        (610, 30.16, 50.0, 1.991), (620, 30.16, 44.0, 2.019), (630, 30.94, 45.3, 2.041),
        (640, 30.46, 53.5, 2.064), (650, 31.74, 60.3, 2.093), (660, 30.12, 42.5, 2.116),
        (670, 31.40, 46.4, 2.137), (680, 30.19, 88.5, 2.174), (690, 29.76, 68.8, 2.206),
        (700, 29.08, 70.7, 2.239), (710, 27.39, 59.2, 2.268), (720, 28.13, 38.9, 2.291),
        (730, 27.81, 69.9, 2.315), (740, 29.70, 43.1, 2.341), (750, 30.19, 53.0, 2.367),
        (760, 30.42, 50.9, 2.391), (770, 27.52, 82.5, 2.426), (780, 28.09, 77.2, 2.461),
        (790, 27.57, 43.3, 2.477), (800, 26.81, 57.9, 2.509),
        (810, 30.50, 60.9, 2.539), (820, 30.18, 46.0, 2.563), (830, 26.58, 77.8, 2.598),
        (840, 26.99, 63.0, 2.629), (850, 25.90, 60.2, 2.660), (860, 27.28, 69.1, 2.685),
        (870, 24.86, 46.3, 2.711), (880, 24.97, 52.7, 2.733), (890, 25.65, 59.8, 2.758),
        (900, 24.91, 73.2, 2.796), (910, 25.38, 39.0, 2.819), (920, 27.47, 46.9, 2.838),
        (930, 25.59, 40.3, 2.856), (940, 25.58, 65.7, 2.875), (950, 26.70, 37.1, 2.903),
        (960, 25.85, 69.3, 2.924), (970, 25.46, 66.7, 2.964), (980, 27.69, 28.6, 2.977),
        (990, 25.86, 41.4, 2.999), (1000, 27.51, 32.1, 3.013),
    ]

    eps = [d[0] for d in log_data]
    rewards = [d[1] for d in log_data]
    costs = [d[2] for d in log_data]
    lambdas = [d[3] for d in log_data]

    # 最佳平衡区间: ep 600-660 (λ≈2.0, Reward≈30, Cost≈44-50)
    sweet_start, sweet_end = 600, 660
    sweet_eps = [e for e in eps if sweet_start <= e <= sweet_end]
    sweet_r = [rewards[i] for i, e in enumerate(eps) if sweet_start <= e <= sweet_end]
    sweet_c = [costs[i] for i, e in enumerate(eps) if sweet_start <= e <= sweet_end]
    sweet_l = [lambdas[i] for i, e in enumerate(eps) if sweet_start <= e <= sweet_end]
    avg_sw_r = np.mean(sweet_r)
    avg_sw_c = np.mean(sweet_c)

    # 前半段 (ep 1-500) 和后半段 (ep 600-1000) 统计
    mid_r_500 = np.mean([rewards[i] for i, e in enumerate(eps) if e <= 500])
    mid_c_500 = np.mean([costs[i] for i, e in enumerate(eps) if e <= 500])
    end_r = np.mean([rewards[i] for i, e in enumerate(eps) if e >= 800])
    end_c = np.mean([costs[i] for i, e in enumerate(eps) if e >= 800])

    fig = plt.figure(figsize=(20, 12))

    # ---- (0,0) 左上：Reward 曲线 + 最佳区间高亮 ----
    ax1 = plt.subplot(2, 3, (1, 2))
    ax1.plot(eps, rewards, color='#3498db', linewidth=1.5, alpha=0.9, label='Avg Reward (per 10 ep)')
    ax1.fill_between(eps, rewards, alpha=0.08, color='#3498db')

    # 高亮最佳平衡区间
    ax1.axvspan(sweet_start, sweet_end, alpha=0.12, color='#2ecc71', label=f'Sweet Spot: ep {sweet_start}-{sweet_end}\n(λ≈2.0, R≈{avg_sw_r:.1f}, C≈{avg_sw_c:.0f})')
    ax1.axhline(y=30, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    # 标注各阶段
    ax1.annotate(f'Phase A: Exploration\nAvg R={mid_r_500:.1f}, C={mid_c_500:.0f}, λ<1.6',
                xy=(250, 31.8), fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f4fd', alpha=0.8))
    ax1.annotate(f'Phase B: Sweet Spot\nAvg R={avg_sw_r:.1f}, C={avg_sw_c:.0f}, λ≈2.0',
                xy=(630, 32.2), fontsize=10, ha='center', fontweight='bold', color='#27ae60',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8faf0', edgecolor='#2ecc71', alpha=0.9))
    ax1.annotate(f'Phase C: Degradation\nAvg R={end_r:.1f}, C={end_c:.0f}, λ>2.5',
                xy=(900, 28.0), fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fde8e8', alpha=0.8))

    ax1.set_ylabel('Average Reward', fontsize=12, color='#3498db')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_title('Phase 4 Reward: 从探索到最佳平衡再到退化', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower left')
    ax1.set_ylim(22, 37)
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ---- (0,2) 右上：Cost 曲线 + 最佳区间高亮 ----
    ax2 = plt.subplot(2, 3, 3)
    ax2.plot(eps, costs, color='#e74c3c', linewidth=1.5, alpha=0.9)
    ax2.fill_between(eps, costs, alpha=0.08, color='#e74c3c')
    ax2.axvspan(sweet_start, sweet_end, alpha=0.12, color='#2ecc71')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax2.axhline(y=avg_sw_c, color='#2ecc71', linestyle='--', alpha=0.5, linewidth=1,
                xmin=(sweet_start-eps[0])/(eps[-1]-eps[0]), xmax=(sweet_end-eps[0])/(eps[-1]-eps[0]))
    ax2.text(sweet_end + 20, avg_sw_c, f'Avg C={avg_sw_c:.0f}', fontsize=8, color='#27ae60', fontweight='bold')

    ax2.set_ylabel('Average Cost', fontsize=12, color='#e74c3c')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_title('Phase 4 Cost: 最佳区间 Cost 被有效压制', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 120)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ---- (1,0) 左下：λ 轨迹 + 2.0 标注 ----
    ax3 = plt.subplot(2, 3, (4, 5))
    ax3.plot(eps, lambdas, color='#9b59b6', linewidth=2, alpha=0.9)
    ax3.fill_between(eps, lambdas, alpha=0.08, color='#9b59b6')
    ax3.axvspan(sweet_start, sweet_end, alpha=0.12, color='#2ecc71')
    ax3.axhline(y=2.0, color='#2ecc71', linestyle='--', alpha=0.6, linewidth=1.5,
                xmin=(sweet_start-eps[0])/(eps[-1]-eps[0]), xmax=(sweet_end-eps[0])/(eps[-1]-eps[0]))
    ax3.annotate('λ = 2.0 (optimal constraint level)',
                xy=(630, 2.0), xytext=(690, 2.3),
                fontsize=10, fontweight='bold', color='#27ae60',
                arrowprops=dict(arrowstyle='->', color='#2ecc71', linewidth=1.5))
    ax3.annotate(f'λ final = {lambdas[-1]:.2f}\n(too strong: reward degrades)',
                xy=(1000, lambdas[-1]), xytext=(850, 2.3),
                fontsize=9, color='#e74c3c', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', linewidth=1))

    ax3.set_ylabel('Lambda (λ)', fontsize=12, color='#9b59b6')
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_title('Phase 4 λ Trajectory: λ=2.0 是最优安全约束水平', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ---- (1,2) 右下：三阶段对比卡片 ----
    ax4 = plt.subplot(2, 3, 6)
    ax4.axis('off')

    phases_info = [
        ('Phase A: Exploration', f'Ep 1-500 | λ: 0.11→1.63', '', ''),
        ('', 'Avg Reward', f'{mid_r_500:.1f}', '#3498db'),
        ('', 'Avg Cost', f'{mid_c_500:.0f}', '#e74c3c'),
        ('', 'Behavior', '高波动, 探索期', '#888'),
        ('', '', '', ''),
        ('Phase B: Sweet Spot', f'Ep 600-660 | λ≈2.0', '', ''),
        ('', 'Avg Reward', f'{avg_sw_r:.1f} ★', '#3498db'),
        ('', 'Avg Cost', f'{avg_sw_c:.0f} ☆', '#e74c3c'),
        ('', 'Behavior', 'Reward保持, Cost最低', '#27ae60'),
        ('', '', '', ''),
        ('Phase C: Degradation', f'Ep 800-1000 | λ: 2.51→3.01', '', ''),
        ('', 'Avg Reward', f'{end_r:.1f} ↓', '#3498db'),
        ('', 'Avg Cost', f'{end_c:.0f}', '#e74c3c'),
        ('', 'Behavior', 'Reward退化, λ过高', '#e74c3c'),
    ]

    y = 0.95
    for label, value, val_color, comment_color in phases_info:
        if label.startswith('Phase'):
            y -= 0.03
            ax4.text(0.05, y, label, fontsize=12, fontweight='bold', color='#2c3e50',
                    transform=ax4.transAxes)
            ax4.plot([0.05, 0.95], [y - 0.025, y - 0.025], linewidth=1.2,
                    color='#ddd', transform=ax4.transAxes, clip_on=False)
            if value:
                ax4.text(0.55, y, value, fontsize=9, color='#888', transform=ax4.transAxes)
            y -= 0.07
        elif label:
            ax4.text(0.12, y, label, fontsize=10, color='#555', fontweight='bold',
                    transform=ax4.transAxes)
            ax4.text(0.55, y, value, fontsize=10, color=val_color, fontweight='bold',
                    transform=ax4.transAxes)
            if comment_color != '#888':
                ax4.text(0.72, y, f'({comment_color})', fontsize=8, color=comment_color,
                        transform=ax4.transAxes)
            y -= 0.06
        else:
            y -= 0.01

    ax4.set_title('Phase 4 Training Stage Summary', fontsize=13, fontweight='bold')

    fig.suptitle('Goal1 Phase 4 详细训练曲线 —— λ=2.0 是最优安全约束平衡点',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'phase4_detailed.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Phase4详细曲线图: {path}")


def plot_v1_failure(data):
    """Goal1 v1 失败训练专项分析图"""
    if 'Goal1 v1(失败)' not in data:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    d_v1 = data['Goal1 v1(失败)']
    segments = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    segment_keys = ['initial_20%', 'early_20-40%', 'mid_40-60%', 'late_60-80%', 'final_20%']
    x = np.arange(len(segments))

    # ---- 左上：Reward 崩塌曲线 ----
    rewards_v1 = [d_v1['segment_analysis'][k]['reward'] for k in segment_keys]
    axes[0, 0].fill_between(x, rewards_v1, alpha=0.15, color='#e74c3c')
    axes[0, 0].plot(x, rewards_v1, 'o-', color='#e74c3c', linewidth=2.5, markersize=10,
                    markerfacecolor='white', markeredgewidth=2)
    for i, (s, r) in enumerate(zip(segments, rewards_v1)):
        color = '#e74c3c' if r < 1 else '#f39c12'
        axes[0, 0].annotate(f'{r:.2f}', (i, r), textcoords="offset points",
                          xytext=(0, 14), ha='center', fontsize=10, fontweight='bold', color=color)
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(segments)
    axes[0, 0].set_title('v1 Reward: 策略崩塌全过程', fontsize=14, fontweight='bold', color='#c0392b')
    axes[0, 0].set_ylabel('Average Reward', fontsize=12)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)

    # ---- 右上：Cost 曲线 ----
    costs_v1 = [d_v1['segment_analysis'][k]['cost'] for k in segment_keys]
    axes[0, 1].fill_between(x, costs_v1, alpha=0.15, color='#e74c3c')
    axes[0, 1].plot(x, costs_v1, 's-', color='#e74c3c', linewidth=2.5, markersize=10,
                    markerfacecolor='white', markeredgewidth=2)
    for i, (s, c) in enumerate(zip(segments, costs_v1)):
        axes[0, 1].annotate(f'{c:.1f}', (i, c), textcoords="offset points",
                          xytext=(0, 14), ha='center', fontsize=10, fontweight='bold', color='#e74c3c')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(segments)
    axes[0, 1].set_title('v1 Cost: 代价波动剧烈', fontsize=14, fontweight='bold', color='#c0392b')
    axes[0, 1].set_ylabel('Average Cost', fontsize=12)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)

    # ---- 左下：λ 瞬间饱和 vs Phase 3 对比 ----
    lambdas_v1 = d_v1['final_statistics']['lambda_trajectory']
    axes[1, 0].plot(range(len(lambdas_v1)), lambdas_v1, color='#e74c3c', linewidth=2,
                    label='Goal1 v1 (Adam, lr=0.01)', alpha=0.9)

    # 标注 λ 饱和点
    axes[1, 0].annotate('λ instantly hits MAX=10.0\nand stays saturated',
                      xy=(50, 10.0), xytext=(len(lambdas_v1)*0.35, 8.5),
                      fontsize=11, fontweight='bold', color='#e74c3c',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0', alpha=0.8),
                      arrowprops=dict(arrowstyle='->', color='#e74c3c', linewidth=1.5))

    # 对比 Phase 3（成功控制 λ）
    if 'Goal1 Phase3' in data:
        lambdas_p3 = data['Goal1 Phase3']['final_statistics']['lambda_trajectory']
        x_p3 = np.linspace(0, len(lambdas_v1), len(lambdas_p3))
        axes[1, 0].plot(x_p3, lambdas_p3, color='#2ecc71', linewidth=2,
                       label='Goal1 Phase3 (SGD, lr=2e-4)', alpha=0.9)
        axes[1, 0].annotate(f'Phase3 λ stays controlled\n(max={lambdas_p3[-1]:.3f})',
                          xy=(x_p3[-1], lambdas_p3[-1]),
                          xytext=(len(lambdas_v1)*0.55, 2.0),
                          fontsize=10, fontweight='bold', color='#2ecc71',
                          arrowprops=dict(arrowstyle='->', color='#2ecc71', linewidth=1.5))

    axes[1, 0].set_title('λ Trajectory: v1 (Adam) vs Phase3 (SGD+EMA+L2)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('PPO Update', fontsize=12)
    axes[1, 0].set_ylabel('Lambda (λ)', fontsize=12)
    axes[1, 0].legend(fontsize=9, loc='center right')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)

    # ---- 右下：v1 vs Phase1 Reward 对比 ----
    if 'Goal1 Phase1' in data:
        d_p1 = data['Goal1 Phase1']
        rewards_p1 = [d_p1['segment_analysis'][k]['reward'] for k in segment_keys]

        bar_width = 0.3
        axes[1, 1].bar(x - bar_width/2, rewards_v1, bar_width, color='#e74c3c', alpha=0.8,
                       label='v1 (failed)', edgecolor='white', linewidth=0.5)
        axes[1, 1].bar(x + bar_width/2, rewards_p1, bar_width, color='#2ecc71', alpha=0.8,
                       label='Phase1 (success)', edgecolor='white', linewidth=0.5)

        # 数值标注
        for i, (rv, rp) in enumerate(zip(rewards_v1, rewards_p1)):
            if rv > 0.1:
                axes[1, 1].text(i - bar_width/2, rv + 1.2, f'{rv:.1f}', ha='center', fontsize=7, color='#e74c3c')
            axes[1, 1].text(i + bar_width/2, rp + 1.2, f'{rp:.1f}', ha='center', fontsize=7, color='#2ecc71')

    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(segments)
    axes[1, 1].set_title('Reward 对比: v1 崩塌 vs Phase1 稳健学习', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Average Reward', fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)

    # 整体标题
    fig.suptitle('Goal1 v1 失败训练深度分析', fontsize=16, fontweight='bold', y=1.01)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'v1_failure_analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] v1失败分析图: {path}")


def main():
    print("=" * 50)
    print("  生成训练曲线图...")
    print("=" * 50)

    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_results()

    if not data:
        print("[错误] 未找到任何训练结果文件")
        return

    print(f"\n已加载 {len(data)} 个训练结果\n")

    plot_goal0_training(data)
    plot_segment_comparison(data)
    plot_reward_cost_evolution(data)
    plot_lambda_trajectories(data)
    plot_phase_comparison(data)
    plot_reward_cost_scatter(data)
    plot_v1_failure(data)
    plot_phase234_combined(data)
    plot_phase1_standalone(data)
    plot_phase4_detailed(data)
    plot_demo_comparison(data)
    plot_training_summary(data)

    print(f"\n全部曲线已保存至: {FIG_DIR}")
    print(f"共生成 {len(os.listdir(FIG_DIR))} 张图片")


if __name__ == '__main__':
    main()
