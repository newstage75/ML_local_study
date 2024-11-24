import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            
        funcs = [self.creator] # List of functions
        while funcs:
            f = funcs.pop() # Get the function
            x, y = f.input, f.output # Get the input and output of the function
            x.grad = f.backward(y.grad) # Call the backward method of the function

            if x.creator is not None:
                funcs.append(x.creator) # Add the parent function to the list
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input): # __call__メソッドは、Pythonの特殊なメソッド。インスタンスを関数のように呼び出せるようにする。
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
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
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)
    
    
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

# A = Square()
# B = Exp()
# C = Square()

x = Variable(np.array(0.5))
# x = Variable(None) #test
# x = Variable(1.0) #test
# a = square(x)
# b = exp(a)
# y = square(b)
y = square(exp(square(x)))

#逆伝播（変数yのbackwardメソッドを呼べば、自動で逆伝搬が行われることを確認）
# y.grad = np.array(1.0) # Simpty set the gradient of the output variable y to 1
y.backward()
print(x.grad)