# 非定常問題
# p31-1
class NonStatBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)
        
    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms) #ノイズを追加
        if rate > np.random.rand():
            return 1
        else:
            return 0

# p35-1
class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha
    
    def update(self, action, reward):
        # alphaで更新
        self.Qs[action] += (reward - self.Qs[action]) + self.alpha
    
    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)
    
# 実行する
import numpy as np
import matplotlib.pyplot as plt # matplotlibのインポート

steps = 1000
epsilon = 0.1
alpha = 0.8

bandit = NonStatBandit()
agent = AlphaAgent(epsilon, alpha)
total_reward = 0
total_rewards = []
rates = []

for step in range(steps):
    action = agent.get_action()  # ①行動を選ぶ
    reward = bandit.play(action) # ②実際にプレイして報酬を得る。
    agent.update(action, reward) # ③行動と報酬から学ぶ
    total_reward += reward
    
    total_rewards.append(total_reward)
    rates.append(total_reward / (step+1))
    
print(total_reward)

# グラフの描画(1)
plt.ylabel('Total reward')
plt.xlabel('Steps')
plt.plot(total_rewards)
plt.show()

# グラフの描画(2)
plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(rates)
plt.show()