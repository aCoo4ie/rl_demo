import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# ========================
# 多臂老虎机环境
# ========================
class Bandit:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.true_rewards = np.random.beta(2, 2, n_arms)
        self.optimal_reward = np.max(self.true_rewards)
        self.optimal_arm = np.argmax(self.true_rewards)
    
    def pull(self, arm):
        return np.random.binomial(1, self.true_rewards[arm])

# ========================
# 四种算法实现
# ========================

def epsilon_greedy_agent(bandit, n_steps, epsilon=0.05, alpha=0.06):
    Q = np.zeros(bandit.n_arms)
    rewards = []
    regrets = []
    total_regret = 0.0
    for t in range(n_steps):
        if np.random.rand() < epsilon:
            a = np.random.randint(bandit.n_arms)
        else:
            a = np.argmax(Q)
        r = bandit.pull(a)
        Q[a] += alpha * (r - Q[a])
        rewards.append(r)
        total_regret += bandit.optimal_reward - bandit.true_rewards[a]
        regrets.append(total_regret)
    return np.array(regrets)

def epsilon_decay_agent(bandit, n_steps, epsilon_max=1.0, epsilon_min=0.01, decay_rate=0.995, alpha=0.1):
    Q = np.zeros(bandit.n_arms)
    rewards = []
    regrets = []
    total_regret = 0.0
    for t in range(n_steps):
        # epsilon = max(epsilon_min, epsilon_max * (decay_rate ** t))
        epsilon = epsilon_min + (epsilon_max - epsilon_min) * (decay_rate ** t)
        if np.random.rand() < epsilon:
            a = np.random.randint(bandit.n_arms)
        else:
            a = np.argmax(Q)
        r = bandit.pull(a)
        Q[a] += alpha * (r - Q[a])
        rewards.append(r)
        total_regret += bandit.optimal_reward - bandit.true_rewards[a]
        regrets.append(total_regret)
    return np.array(regrets)

def ucb_agent(bandit, n_steps, c=1.0, alpha=0.1):
    Q = np.zeros(bandit.n_arms)
    N = np.zeros(bandit.n_arms)
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
        total_regret += bandit.optimal_reward - bandit.true_rewards[a]
        regrets.append(total_regret)
    return np.array(regrets)

def thompson_sampling_agent(bandit, n_steps, alpha0=1, beta0=1):
    alpha = np.full(bandit.n_arms, alpha0)
    beta = np.full(bandit.n_arms, beta0)
    regrets = []
    total_regret = 0.0
    for t in range(n_steps):
        # 从 Beta 分布采样
        theta = np.random.beta(alpha, beta)
        a = np.argmax(theta)
        r = bandit.pull(a)
        # 贝叶斯更新
        if r == 1:
            alpha[a] += 1
        else:
            beta[a] += 1
        total_regret += bandit.optimal_reward - bandit.true_rewards[a]
        regrets.append(total_regret)
    return np.array(regrets)

# ========================
# 实验设置
# ========================
n_arms = 5
n_steps = 100000
n_runs = 100

algorithms = {
    'ε-greedy': epsilon_greedy_agent,
    'ε-decay': epsilon_decay_agent,
    'UCB': ucb_agent,
    'Thompson': thompson_sampling_agent
}

all_regrets = {name: [] for name in algorithms}

print("开始实验...")

for run in range(n_runs):
    if (run + 1) % 5 == 0:
        print(f"运行 {run + 1}/{n_runs}")
    bandit = Bandit(n_arms=n_arms)
    for name, agent in algorithms.items():
        regrets = agent(bandit, n_steps)
        all_regrets[name].append(regrets)

# ========================
# 计算平均累积懊悔
# ========================
mean_regrets = {}
for name, regrets_list in all_regrets.items():
    mean_regrets[name] = np.mean(regrets_list, axis=0)

# ========================
# 绘图：验证线性 vs 对数增长
# ========================
plt.figure(figsize=(12, 8))

steps = np.arange(1, n_steps + 1)

# 使用对数 x 轴
plt.semilogx(steps, mean_regrets['ε-greedy'], label='ε-greedy (linear)', linewidth=2)
plt.semilogx(steps, mean_regrets['ε-decay'], label='ε-decay (log)', linewidth=2)
plt.semilogx(steps, mean_regrets['UCB'], label='UCB (log)', linewidth=2)
plt.semilogx(steps, mean_regrets['Thompson'], label='Thompson Sampling (log)', linewidth=2)

plt.xlabel('Time Step (log scale)')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret: Linear vs Logarithmic Growth')
plt.legend()
plt.grid(True, which="both", alpha=0.3)

# 添加理论趋势线（可选）
T = steps[-1]
log_T = np.log(steps) * 50  # 缩放以便比较
plt.semilogx(steps, 0.1 * steps, 'k--', alpha=0.5, label='Theoretical O(T)', linewidth=2)
plt.semilogx(steps, 50 * np.log(steps), 'k:', alpha=0.5, label='Theoretical O(log T)', linewidth=2)

plt.tight_layout()
plt.show()

# ========================
# 打印最终增长率
# ========================
final_T = n_steps
print(f"\n在 T = {final_T} 时的累积懊悔：")
for name, regret in mean_regrets.items():
    rate = regret[-1] / final_T
    # if 'greedy' in name:
    #     print(f"{name}: {regret[-1]:.2f} → 线性趋势 (R/T = {rate:.4f})")
    # else:
    log_rate = regret[-1] / np.log(final_T)
    print(f"{name}: {regret[-1]:.2f} → 对数趋势 (R/log T = {log_rate:.2f})")