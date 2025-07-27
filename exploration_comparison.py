import numpy as np
import matplotlib.pyplot as plt

# ========================
# 多臂老虎机实验：对比三种探索策略
# ========================

class MultiArmedBandit:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        # 真实奖励概率：每个臂的胜率（伯努利分布）
        self.true_rewards = np.random.beta(2, 2, n_arms)
        self.best_reward = np.max(self.true_rewards)
        self.optimal_arm = np.argmax(self.true_rewards)
    
    def pull(self, arm):
        """拉动第 arm 个臂，返回奖励（0 或 1）"""
        return np.random.binomial(1, self.true_rewards[arm])


def epsilon_greedy_agent(bandit, n_steps, epsilon=0.1, alpha=0.1):
    Q = np.zeros(bandit.n_arms)  # 动作价值估计
    N = np.zeros(bandit.n_arms)  # 各臂尝试次数
    rewards = []
    regrets = []
    total_regret = 0.0
    
    for t in range(n_steps):
        if np.random.rand() < epsilon:
            a = np.random.randint(bandit.n_arms)
        else:
            a = np.argmax(Q)
        
        r = bandit.pull(a)
        N[a] += 1
        Q[a] += alpha * (r - Q[a])  # 增量更新
        
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
            a = t - 1  # 先每个臂试一次
        else:
            # UCB 公式：Q + c * sqrt(ln(t)/N)
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
    Q = np.zeros(bandit.n_arms)  # 真实价值学习
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

n_arms = 10
n_steps = 8000
n_runs = 100  # 多次运行取平均

# 存储结果
all_rewards = {'epsilon_greedy': [], 'ucb': [], 'noisy_nets': []}
all_regrets = {'epsilon_greedy': [], 'ucb': [], 'noisy_nets': []}

print("开始实验...")

for run in range(n_runs):
    if (run + 1) % 10 == 0:
        print(f"运行 {run + 1}/{n_runs}")
    
    bandit = MultiArmedBandit(n_arms=n_arms)
    
    # 运行三种策略
    r1, reg1 = epsilon_greedy_agent(bandit, n_steps, epsilon=0.1)
    r2, reg2 = ucb_agent(bandit, n_steps, c=2.0)
    r3, reg3 = noisy_nets_agent(bandit, n_steps, noise_scale=0.5)
    
    all_rewards['epsilon_greedy'].append(r1)
    all_rewards['ucb'].append(r2)
    all_rewards['noisy_nets'].append(r3)
    
    all_regrets['epsilon_greedy'].append(reg1)
    all_regrets['ucb'].append(reg2)
    all_regrets['noisy_nets'].append(reg3)

# ========================
# 计算平均值和置信区间
# ========================

def compute_mean_std(data):
    arr = np.array(data)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return mean, std

mean_r_eg, std_r_eg = compute_mean_std(all_rewards['epsilon_greedy'])
mean_r_ucb, std_r_ucb = compute_mean_std(all_rewards['ucb'])
mean_r_noisy, std_r_noisy = compute_mean_std(all_rewards['noisy_nets'])

mean_reg_eg, _ = compute_mean_std(all_regrets['epsilon_greedy'])
mean_reg_ucb, _ = compute_mean_std(all_regrets['ucb'])
mean_reg_noisy, _ = compute_mean_std(all_regrets['noisy_nets'])

# ========================
# 绘图
# ========================

steps = np.arange(n_steps)

plt.figure(figsize=(14, 6))

# 子图 1：平均奖励
plt.subplot(1, 2, 1)
plt.plot(steps, mean_r_eg, label='ε-greedy (ε=0.1)', linewidth=2)
plt.fill_between(steps, mean_r_eg - std_r_eg/np.sqrt(n_runs), mean_r_eg + std_r_eg/np.sqrt(n_runs), alpha=0.2)

plt.plot(steps, mean_r_ucb, label='UCB (c=2.0)', linewidth=2)
plt.plot(steps, mean_r_noisy, label='Noisy Nets (σ=0.5)', linewidth=2)

plt.xlabel('Time Step')
plt.ylabel('Average Reward')
plt.title('Exploration Strategy Comparison: Average Reward')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图 2：累计后悔（越低越好）
plt.subplot(1, 2, 2)
plt.plot(steps, mean_reg_eg, label='ε-greedy', linewidth=2)
plt.plot(steps, mean_reg_ucb, label='UCB', linewidth=2)
plt.plot(steps, mean_reg_noisy, label='Noisy Nets', linewidth=2)

plt.xlabel('Time Step')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================
# 打印最终性能
# ========================

print("\n最终平均性能（最后100步）：")
print(f"ε-greedy:     {mean_r_eg[-100:].mean():.3f} ± {std_r_eg[-100:].mean()/np.sqrt(n_runs):.3f}")
print(f"UCB:          {mean_r_ucb[-100:].mean():.3f} ± {std_r_ucb[-100:].mean()/np.sqrt(n_runs):.3f}")
print(f"Noisy Nets:   {mean_r_noisy[-100:].mean():.3f} ± {std_r_noisy[-100:].mean()/np.sqrt(n_runs):.3f}")

print(f"\n最优臂真实奖励: {np.max(bandit.true_rewards):.3f}")