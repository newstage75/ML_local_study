import numpy as np

c = np.array([[1, 0, 0, 0, 0, 0, 0]]) #入力
W = np.random.randn(7,3) #重み
h = np.dot(c, W) #中間ノード
print(h)
print("Wの重みは", W)
print("cの次元は", c.ndim)
print("hの次元は", h.ndim)
