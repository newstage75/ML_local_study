import numpy as np

class Bandit:
    def __init__(self,arms=10):
        self.rates = np.random.rand(arms) #各マシンの確率
    
    def play(self, arm):
        rate =self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0
        
# デバッグ用
# print(np.random.rand(10)) 
#[0.77372433 0.8525301  0.08218402 0.17681752 0.53832741 0.47010984 0.15247564 0.01923318 0.71152829 0.1356219 ]
#つまり、10台のスロットマシンの勝率をランダムに生成

# play(self, arm):　は何台目のスロットマシンを遊ぶかを想定したもの

bandit = Bandit()

# p21-1
# for i in range(3):
#     print(bandit.play(0),'-',bandit.rates[i])

# p21-2
# Q = 0
# for n in range(1,11):
#     reward = bandit.play(0) #0番目のスロットマシンをプレイ
#     Q += (reward - Q)/n
#     print(f'{n}番目の推定値：{Q}')
#     print(bandit.rates[0]) #初期化で各マシンの確率を出しているので、数値は変わらない


# p21-3
# Qs = np.zeros(10)
# ns = np.zeros(10)
# print(Qs)

# for n in range(10):
#     action = np.random.randint(0,10) # ランダムな行動
#     reward = bandit.play(action)
    
#     ns[action] += 1
#     Qs[action] += (reward - Qs[action]) / ns[action]
#     print(Qs)
    
# 実行結果（一例）
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
# [0.  0.  0.  0.5 0.  0.  0.  0.  1.  0. ]
# [0.  0.  0.  0.5 0.  0.  0.  0.  1.  0. ]
# [0.  0.  1.  0.5 0.  0.  0.  0.  1.  0. ]
# [0.  0.  1.  0.5 0.  0.  0.  0.  1.  0. ]
# [0.  0.  1.  0.5 0.  0.  0.  0.  1.  0. ]
# [0.  0.  1.  0.5 0.  0.  0.  0.  1.  0. ]
    

# p22-1
# 以上を踏まえた上で、Agentクラスの実装（ε-greed法）
class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)
        
        
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]
        
    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)
    
# 初期値のepsilonはε-greedy法におけるランダムに行動する確率。
# action_sizeはエージェントが選択できる行動の数。
# バンディット問題の場合、action_seizeはスロットマシンの台数に該当。
# 最後のget_actionで、self.epsilon以下の値が出たら、ランダムでスロットマシンを活用。

# argmax の意味
# 例 x = [2, 5, 1] だと
# max(x) = 5（最大値）
# argmax(x) = 1（5 は インデックス 1 にある）

# Agentクラスをテストしてみる（独習）
# agent = Agent(0.1, 10)
# for i in range(10):
#     action = agent.get_action()
#     reward = bandit.play(action)
#     agent.update(action, reward)
# print(agent.Qs) #実行結果の１例 [0.75 0.   1.   0.   0.   0.   0.   0.   0.   0.  ]