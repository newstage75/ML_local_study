import weakref
import numpy as np
import contextlib # contextlibを追加

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
            
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 # generationの初期化
    
    def cleargrad(self):
        self.grad = None
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 #　親世代+1
        
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            
        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
                
        add_func(self.creator)
        
        while funcs:
            f = funcs.pop()
            gyx = [output().grad for output in f.outputs] # 出力変数であるoutputsのgrad（微分）をリストにまとめる
            gxs = f.backward(*gyx) # 関数fの逆伝搬を呼び出す
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    add_func(x.creator)
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # yはweakref。デフォルトでretain_grad=Falseの場合、途中の変数の微分はリセットされる。

                
def as_array(x):
    if np.isscalar(x): #スカラーの場合はnp.array()である多次元配列の形式に変換する。
        return np.array(x)
    return x

class Config:
    enable_backprop = True # Trueの場合は逆伝搬を行う。Falseの場合は逆伝搬を行わない。

class Function:
    #__call__メソッドで出力変数に対して、いくつかの設定を行う
    def __call__(self, *inputs): # *inputs（頭にアスタリスク）で可変長引数を受け取る
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # *xs（頭にアスタリスク）でリストの要素を展開して渡す（アンパッキング）
        if not isinstance(ys, tuple): #タプルでない場合はタプルに変換
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs]) # 世代の設定
            for output in outputs:
                output.set_creator(self) # つながりの設定
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            
        
        # リストの要素が1つの場合はその要素を返す
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self,xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        # print('Add backward gy:', gy) # gyは1.0？
        return gx
    

def square(x):
    return Square()(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y)
    
    def backward(self, gy): #上流の微分を"そのまま流す"のが足し算の逆伝搬
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)

def no_grad():
    return using_config('enable_backprop', False)

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)