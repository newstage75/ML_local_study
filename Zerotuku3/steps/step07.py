import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        f = self.creator # Get the function which created this variable
        if f is not None:
            x = f.input # Get the input of the function
            x.grad = f.backward(self.grad) # Call the backward method of the function
            x.backward() # Call the backward method of the input variable(recursive：再帰)


class Function:
    def __call__(self, input): # __call__メソッドは、Pythonの特殊なメソッド。インスタンスを関数のように呼び出せるようにする。
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) # Set parent function
        self.input = input # Remember the input variable
        self.output = output
        return output
    
    def forward(self,x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
   
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
    
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

#逆向きに計算グラフのノードを辿る（テスト）
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

#逆伝播（変数yのbackwardメソッドを呼べば、自動で逆伝搬が行われることを確認）
y.grad = np.array(1.0)
y.backward()
print(x.grad)