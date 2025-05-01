import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data  # 儲存該值
        self._children = set(_children)  # 儲存該值的父節點（如果有的話）
        self._op = _op  # 儲存操作符
        self.grad = 0.0  # 儲存該值的梯度

    def sigmoid(self):
        # Sigmoid 函數的實現
        out = Value(1 / (1 + math.exp(-self.data)), (self,), 'sigmoid')
        
        # 在反向傳播時計算梯度
        def sigmoid_grad(grad):
            return grad * (1 - out.data) * out.data
        
        out._backward = sigmoid_grad
        return out

    def exp(self):
        # Exp 函數的實現
        out = Value(math.exp(self.data), (self,), 'exp')

        # 在反向傳播時計算梯度
        def exp_grad(grad):
            return grad * out.data
        
        out._backward = exp_grad
        return out
