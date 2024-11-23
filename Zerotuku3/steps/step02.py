import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self,x):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2


# __call__メソッドは、Pythonの特殊なメソッド。
# このメソッドを定義すれば、f=Function()としたとき、f(...)と書くことで、__call__メソッドを呼び出せる。

# メソッドforward です。このメソッドは、引数 x を受け取りますが、実際の処理内容はまだ実装されていません。
# その代わりに、NotImplementedError 例外を発生させることで、サブクラスでこのメソッドをオーバーライドして実装する必要があることを示しています。
# これは、抽象メソッドのような役割を果たします。

x = Variable(np.array(10))
f = Square()
y = f(x)

print(type(y))
print(y.data)