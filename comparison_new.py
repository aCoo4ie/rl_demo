import numpy as np
import matplotlib.pyplot as plt

# ========================
# 多臂老虎机实验：对比三种探索策略（增强版）
# 改进：Noisy Nets 噪声调参、ε-decay、平滑曲线、结果表格
# ========================

class MultiArmedBandit:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.true_rewards = np.random.beta(2, 2, n_arms)
        self.best_reward = np.max(self.true_rewards)
        self.optimal_arm = np.argmax(self.true_rewards)
    
    def pull(self, arm):
        return np.random.binomial(1, self.true_rewards[arm])


def epsilon_greedy_decay_agent(bandit, n_steps, epsilon_max=1.0, epsilon_min=0.01, decay_rate=0.995, alpha=0.1):
    Q = np.zeros(bandit.n_arms)
    N = np.zeros(bandit.n_arms)
    rewards = []
    regrets = []
    total_regret = 0.0
    
    for t in range(n_steps):
        epsilon = max(epsilon_min, epsilon_max * (decay_rate ** t))
        if np.random.rand() < epsilon:
            a = np.random.randint(bandit.n_arms)
        else:
            a = np.argmax(Q)
        
        r = bandit.pull(a)
        N[a] += 1
        Q[a] += alpha * (r - Q[a])
        
        rewards.append(r)
        total_regret += bandit.best_reward - bandit.true_rewards[a]
        regrets.append(total_regret)
    
    return np.array(rewards), np.array(regrets)


def ucb_agent(bandit, n_steps, c=2.0, alpha=0.1):
    Q = np.zeros(bandit.n_arms)
    N = np.zeros(bandit.n_arms)
    rewards = []
    regrets = []
    total_regret = 0.0
    
    for t in range(1, n_steps + 1):
        if t <= bandit.n_arms:
            a = t - 1
        else:
            upper_bounds = Q + c * np.sqrt(np.log(t) / (N + 1e-8))
            a = np.argmax(upper_bounds)
        
        r = bandit.pull(a)
        N[a] += 1
        Q[a] += alpha * (r - Q[a])
        
        rewards.append(r)
        total_regret += bandit.best_reward - bandit.true_rewards[a]
        regrets.append(total_regret)
    
    return np.array(rewards), np.array(regrets)


def noisy_nets_agent(bandit, n_steps, noise_scale=0.5, alpha=0.1):
    Q = np.zeros(bandit.n_arms)
    rewards = []
    regrets = []
    total_regret = 0.0
    
    for t in range(n_steps):
        noise = np.random.normal(0, noise_scale, bandit.n_arms)
        Q_noisy = Q + noise
        a = np.argmax(Q_noisy)
        
        r = bandit.pull(a)
        Q[a] += alpha * (r - Q[a])
        
        rewards.append(r)
        total_regret += bandit.best_reward - bandit.true_rewards[a]
        regrets.append(total_regret)
    
    return np.array(rewards), np.array(regrets)


# ========================
# 实验设置
# ========================
n_arms = 20
n_steps = 100000
n_runs = 100

# 存储结果
all_rewards = {'epsilon_decay': [], 'ucb': [], 'noisy_nets_low': [], 'noisy_nets_high': []}
all_regrets = {'epsilon_decay': [], 'ucb': [], 'noisy_nets_low': [], 'noisy_nets_high': []}

print("开始实验...")

for run in range(n_runs):
    if (run + 1) % 10 == 0:
        print(f"运行 {run + 1}/{n_runs}")
    
    bandit = MultiArmedBandit(n_arms=n_arms)
    
    # 运行四种策略
    r1, reg1 = epsilon_greedy_decay_agent(bandit, n_steps)
    r2, reg2 = ucb_agent(bandit, n_steps)
    r3, reg3 = noisy_nets_agent(bandit, n_steps, noise_scale=0.3)  # 低噪声
    r4, reg4 = noisy_nets_agent(bandit, n_steps, noise_scale=0.8)  # 高噪声
    
    all_rewards['epsilon_decay'].append(r1)
    all_rewards['ucb'].append(r2)
    all_rewards['noisy_nets_low'].append(r3)
    all_rewards['noisy_nets_high'].append(r4)
    
    all_regrets['epsilon_decay'].append(reg1)
    all_regrets['ucb'].append(reg2)
    all_regrets['noisy_nets_low'].append(reg3)
    all_regrets['noisy_nets_high'].append(reg4)


# ========================
# 计算平均值和标准误差
# ========================
def compute_mean_std(data):
    arr = np.array(data)
    mean = np.mean(arr, axis=0)
    sem = np.std(arr, axis=0) / np.sqrt(len(data))  # 标准误差
    return mean, sem

mean_r_eg, sem_r_eg = compute_mean_std(all_rewards['epsilon_decay'])
mean_r_ucb, sem_r_ucb = compute_mean_std(all_rewards['ucb'])
mean_r_noisy_low, sem_r_noisy_low = compute_mean_std(all_rewards['noisy_nets_low'])
mean_r_noisy_high, sem_r_noisy_high = compute_mean_std(all_rewards['noisy_nets_high'])

mean_reg_eg, _ = compute_mean_std(all_regrets['epsilon_decay'])
mean_reg_ucb, _ = compute_mean_std(all_regrets['ucb'])
mean_reg_noisy_low, _ = compute_mean_std(all_regrets['noisy_nets_low'])
mean_reg_noisy_high, _ = compute_mean_std(all_regrets['noisy_nets_high'])


# ========================
# 平滑函数（滑动窗口均值）
# ========================
def smooth(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')

# 平滑奖励曲线
smooth_eg = smooth(mean_r_eg)
smooth_ucb = smooth(mean_r_ucb)
smooth_noisy_low = smooth(mean_r_noisy_low)
smooth_noisy_high = smooth(mean_r_noisy_high)

# 平滑起始步数
smooth_steps = np.arange(len(smooth_eg)) + (50 - 1)//2


# ========================
# 绘图
# ========================
plt.figure(figsize=(16, 6))

# 子图 1：平滑后的平均奖励
plt.subplot(1, 2, 1)
plt.plot(smooth_steps, smooth_eg, label='ε-greedy (decay)', linewidth=2)
plt.fill_between(smooth_steps, smooth(mean_r_eg - sem_r_eg), smooth(mean_r_eg + sem_r_eg), alpha=0.2)

plt.plot(smooth_steps, smooth_ucb, label='UCB (c=2.0)', linewidth=2)
plt.fill_between(smooth_steps, smooth(mean_r_ucb - sem_r_ucb), smooth(mean_r_ucb + sem_r_ucb), alpha=0.2)

plt.plot(smooth_steps, smooth_noisy_low, label='Noisy Nets (σ=0.3)', linewidth=2)
plt.fill_between(smooth_steps, smooth(mean_r_noisy_low - sem_r_noisy_low), smooth(mean_r_noisy_low + sem_r_noisy_low), alpha=0.2)

plt.plot(smooth_steps, smooth_noisy_high, label='Noisy Nets (σ=0.8)', linewidth=2)
plt.fill_between(smooth_steps, smooth(mean_r_noisy_high - sem_r_noisy_high), smooth(mean_r_noisy_high + sem_r_noisy_high), alpha=0.2)

plt.xlabel('Time Step')
plt.ylabel('Smoothed Average Reward')
plt.title('Exploration Strategies: Smoothed Average Reward')
plt.legend()
plt.grid(True, alpha=0.3)


# 子图 2：累计后悔
plt.subplot(1, 2, 2)
plt.plot(mean_reg_eg, label='ε-greedy (decay)', linewidth=2)
plt.plot(mean_reg_ucb, label='UCB', linewidth=2)
plt.plot(mean_reg_noisy_low, label='Noisy Nets (σ=0.3)', linewidth=2)
plt.plot(mean_reg_noisy_high, label='Noisy Nets (σ=0.8)', linewidth=2, linestyle='--')

plt.xlabel('Time Step')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret Over Time')
plt.legend()
plt.grid(True, alpha=0.3)


# ========================
# 添加性能对比表格
# ========================
final_step = -100
final_rewards = {
    'Strategy': ['ε-greedy (decay)', 'UCB', 'Noisy Nets (σ=0.3)', 'Noisy Nets (σ=0.8)'],
    'Mean Reward': [
        f"{mean_r_eg[final_step:].mean():.3f}",
        f"{mean_r_ucb[final_step:].mean():.3f}",
        f"{mean_r_noisy_low[final_step:].mean():.3f}",
        f"{mean_r_noisy_high[final_step:].mean():.3f}"
    ],
    'Regret Rate': [
        f"{mean_reg_eg[-1]/n_steps:.3f}",
        f"{mean_reg_ucb[-1]/n_steps:.3f}",
        f"{mean_reg_noisy_low[-1]/n_steps:.3f}",
        f"{mean_reg_noisy_high[-1]/n_steps:.3f}"
    ]
}

# 在图右侧添加表格
# plt.figtext(0.92, 0.6, "Final Performance (Last 100 Steps)", fontsize=12, fontweight='bold')
# table_str = "Strategy Reward  Regret/step\n" + "-"*40 + "\n"
# for i in range(len(final_rewards['Strategy'])):
#     table_str += f"{final_rewards['Strategy'][i]:<20}  {final_rewards['Mean Reward'][i]:>6}  {final_rewards['Regret Rate'][i]:>6}\n"

# plt.figtext(0.92, 0.1, table_str, fontsize=10, fontfamily='monospace', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))


# ========================
# 调整布局并显示
# ========================
plt.tight_layout(rect=[0, 0, 0.88, 1])  # 留出右侧空间给表格
plt.show()


# ========================
# 打印最终结果
# ========================
print("\n最终平均性能：")
print(f"ε-greedy (decay):     {mean_r_eg[:].mean():.3f} ± {sem_r_eg[:].mean():.3f}")
print(f"UCB:                  {mean_r_ucb[:].mean():.3f} ± {sem_r_ucb[:].mean():.3f}")
print(f"Noisy Nets (σ=0.3):   {mean_r_noisy_low[:].mean():.3f} ± {sem_r_noisy_low[:].mean():.3f}")
print(f"Noisy Nets (σ=0.8):   {mean_r_noisy_high[:].mean():.3f} ± {sem_r_noisy_high[:].mean():.3f}")

print(f"\n最优臂真实奖励: {np.max(bandit.true_rewards):.3f}")


# ========================
# 封装实验主流程
# ========================
def run_experiment(n_arms, n_steps, n_runs=30, save_fig=True):
    all_rewards = {'epsilon_decay': [], 'ucb': [], 'noisy_nets_low': [], 'noisy_nets_high': []}
    all_regrets = {'epsilon_decay': [], 'ucb': [], 'noisy_nets_low': [], 'noisy_nets_high': []}

    print(f"\n开始实验: n_arms={n_arms}, n_steps={n_steps} ...")
    for run in range(n_runs):
        if (run + 1) % 10 == 0:
            print(f"运行 {run + 1}/{n_runs}")
        bandit = MultiArmedBandit(n_arms=n_arms)
        r1, reg1 = epsilon_greedy_decay_agent(bandit, n_steps)
        r2, reg2 = ucb_agent(bandit, n_steps)
        r3, reg3 = noisy_nets_agent(bandit, n_steps, noise_scale=0.3)
        r4, reg4 = noisy_nets_agent(bandit, n_steps, noise_scale=0.8)
        all_rewards['epsilon_decay'].append(r1)
        all_rewards['ucb'].append(r2)
        all_rewards['noisy_nets_low'].append(r3)
        all_rewards['noisy_nets_high'].append(r4)
        all_regrets['epsilon_decay'].append(reg1)
        all_regrets['ucb'].append(reg2)
        all_regrets['noisy_nets_low'].append(reg3)
        all_regrets['noisy_nets_high'].append(reg4)

    def compute_mean_std(data):
        arr = np.array(data)
        mean = np.mean(arr, axis=0)
        sem = np.std(arr, axis=0) / np.sqrt(len(data))
        return mean, sem

    mean_r_eg, sem_r_eg = compute_mean_std(all_rewards['epsilon_decay'])
    mean_r_ucb, sem_r_ucb = compute_mean_std(all_rewards['ucb'])
    mean_r_noisy_low, sem_r_noisy_low = compute_mean_std(all_rewards['noisy_nets_low'])
    mean_r_noisy_high, sem_r_noisy_high = compute_mean_std(all_rewards['noisy_nets_high'])
    mean_reg_eg, _ = compute_mean_std(all_regrets['epsilon_decay'])
    mean_reg_ucb, _ = compute_mean_std(all_regrets['ucb'])
    mean_reg_noisy_low, _ = compute_mean_std(all_regrets['noisy_nets_low'])
    mean_reg_noisy_high, _ = compute_mean_std(all_regrets['noisy_nets_high'])

    def smooth(x, window=50):
        return np.convolve(x, np.ones(window)/window, mode='valid')

    smooth_eg = smooth(mean_r_eg)
    smooth_ucb = smooth(mean_r_ucb)
    smooth_noisy_low = smooth(mean_r_noisy_low)
    smooth_noisy_high = smooth(mean_r_noisy_high)
    smooth_steps = np.arange(len(smooth_eg)) + (50 - 1)//2

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(smooth_steps, smooth_eg, label='ε-greedy (decay)', linewidth=2)
    plt.fill_between(smooth_steps, smooth(mean_r_eg - sem_r_eg), smooth(mean_r_eg + sem_r_eg), alpha=0.2)
    plt.plot(smooth_steps, smooth_ucb, label='UCB (c=2.0)', linewidth=2)
    plt.fill_between(smooth_steps, smooth(mean_r_ucb - sem_r_ucb), smooth(mean_r_ucb + sem_r_ucb), alpha=0.2)
    plt.plot(smooth_steps, smooth_noisy_low, label='Noisy Nets (σ=0.3)', linewidth=2)
    plt.fill_between(smooth_steps, smooth(mean_r_noisy_low - sem_r_noisy_low), smooth(mean_r_noisy_low + sem_r_noisy_low), alpha=0.2)
    plt.plot(smooth_steps, smooth_noisy_high, label='Noisy Nets (σ=0.8)', linewidth=2)
    plt.fill_between(smooth_steps, smooth(mean_r_noisy_high - sem_r_noisy_high), smooth(mean_r_noisy_high + sem_r_noisy_high), alpha=0.2)
    plt.xlabel('Time Step')
    plt.ylabel('Smoothed Average Reward')
    plt.title(f'Smoothed Avg Reward (n_arms={n_arms}, n_steps={n_steps})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(mean_reg_eg, label='ε-greedy (decay)', linewidth=2)
    plt.plot(mean_reg_ucb, label='UCB', linewidth=2)
    plt.plot(mean_reg_noisy_low, label='Noisy Nets (σ=0.3)', linewidth=2)
    plt.plot(mean_reg_noisy_high, label='Noisy Nets (σ=0.8)', linewidth=2, linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Regret')
    plt.title(f'Cumulative Regret (n_arms={n_arms}, n_steps={n_steps})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 1])
    if save_fig:
        fig_name = f"result_narms{n_arms}_nsteps{n_steps}.png"
        plt.savefig(fig_name)
        print(f"图已保存: {fig_name}")
    else:
        plt.show()
    plt.close()

# 主循环：遍历参数组合
if __name__ == "__main__":
    for n_arms in range(10, 101, 10):
        for n_steps in range(5000, 10001, 1000):
            run_experiment(n_arms, n_steps, n_runs=30, save_fig=True)