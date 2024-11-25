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
    if np.isscalar(x): #スカラーの場合はnp.array()である多次元配列の形式に変換する。
        return np.array(x)
    return x

class Function:
    #__call__メソッドで出力変数に対して、いくつかの設定を行う
    def __call__(self, *inputs): # *inputs（頭にアスタリスク）で可変長引数を受け取る
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # *xs（頭にアスタリスク）でリストの要素を展開して渡す（アンパッキング）
        if not isinstance(ys, tuple): #タプルでない場合はタプルに変換
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        # 生成元となる関数（self）を設定している。
        # set_creatorメソッドを呼び出すことで、逆伝搬の際にどの関数がこの変数を生成したかを追跡できる。
        for output in outputs:
            output.set_creator(self)
        # 入力と出力をインスタンス変数として保存
        self.inputs = inputs
        self.outputs = outputs
        # リストの要素が1つの場合はその要素を返す
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self,xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y)
    
def add(x0, x1):
    return Add()(x0, x1)
    
x1 = Variable(np.array(2))
x2 = Variable(np.array(3))
y = add(x1, x2)
print(y.data)