if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import math
from dezero import Variable, Function
from dezero.utils import plot_dot_graph

# Sin関数の実装（解析的に解く）
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

x = Variable(np.array(np.pi/4))
y = sin(x)
y.backward()

print('--- original sin ---')
print(y.data)
print(x.grad)
print(1/np.sqrt(2))


# テイラー展開によるsin関数の実装
# math.factorialは階乗を計算する関数
def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1)**i / math.factorial(2*i+1)
        t = c*x**(2*i+1)
        y = y + t
        # tの絶対値がthreshold未満になったら終了
        if abs(t.data) < threshold:
            break
    return y

x = Variable(np.array(np.pi / 4))
y = my_sin(x)  # , threshold=1e-150)
y.backward()
print('--- approximate sin ---')
print(y.data)
print(x.grad)

x.name = 'x'
y.name = 'y'
plot_dot_graph(y, verbose=False, to_file='my_sin.png')