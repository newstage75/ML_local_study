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