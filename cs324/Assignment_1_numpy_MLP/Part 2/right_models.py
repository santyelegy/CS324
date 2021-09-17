import numpy as np

class Softmax():
    def forward(self, x):
        '''
        x.shape = (N, C)
        接收批量的输入，每个输入是一维向量
        计算公式为：
        a_{ij}=\frac{e^{x_{ij}}}{\sum_{j}^{C} e^{x_{ij}}}
        '''
        v = np.exp(x - x.max(axis=-1, keepdims=True))    
        return v / v.sum(axis=-1, keepdims=True)
    
    def backward(self, y):
        # 一般Softmax的反向传播和CrossEntropyLoss的放在一起
        pass


class True_Relu():
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, eta):
        eta[self.x<=0] = 0
        return eta


#交叉熵损失函数
class CrossEntropyLoss(object):
    def __init__(self):
        # 内置一个softmax作为分类器
        self.classifier = Softmax()

    def gradient(self):
        return self.grad

    def __call__(self, a, y, requires_acc=True):
        '''
        a: 批量的样本输出
        y: 批量的样本真值
        requires_acc: 是否输出正确率
        return: 该批样本的平均损失[, 正确率]
        输出与真值的shape是一样的，并且都是批量的，单个输出与真值是一维向量
        a.shape = y.shape = (N, C)      N是该批样本的数量，C是单个样本最终输出向量的长度
        '''
        # 网络的输出不应该经过softmax分类，而在交叉熵损失函数中进行
        a = self.classifier.forward(a)
        # 提前计算好梯度
        self.grad = a - y
        # 样本整体损失
        # L_{i}=-\sum_{j}^{C} y_{ij} \ln a_{ij}
        # 样本的平均损失
        # L_{mean}=\frac{1}{N} \sum_{i}^{N} L_{i}=-\frac{1}{N} \sum_{i}^{N} \sum_{j}^{C} y_{ij} \ln a_{ij}
        loss = -1 * np.einsum('ij,ij->', y, np.log(a), optimize=True) / y.shape[0]
        if requires_acc:
            acc = np.argmax(a, axis=-1) == np.argmax(y, axis=-1)
            return acc.mean(), loss
        return loss


class True_Linear():
    def __init__(self, shape, requires_grad=True, bias=True, **kwargs):
        '''
        shape: (in_size, out_size)
        requires_grad: 是否在反向传播中计算权重梯度
        bias: 是否设置偏置
        '''
        self.W = np.random.normal(0,0.0001,(shape[0],shape[1]))
        self.b = np.zeros(shape[-1]) if bias else None
        self.require_grad = requires_grad

    def forward(self, x):
        if self.require_grad: self.x = x
        # 公式：a_{ik}=\sum_{j}^{C} x_{ij} w_{jk}
        a = np.dot(x, self.W.data)
        if self.b is not None: a += self.b
        return a

    def backward(self, eta):
        # 在反向计算中矩阵乘法涉及转置，einsum比dot稍好一点点
        if self.require_grad:
            batch_size = eta.shape[0]
            # 公式：dW_{ik}=\frac {1}{N} \sum_{j}^{C} x_{ji} da_{jk}
            self.W_grad = np.einsum('ji,jk->ik', self.x, eta)
            # 公式：db_{*}=\frac {1}{N} \sum_{i}^{N} da_{i*}
            if self.b is not None: self.b_grad = np.einsum('i...->...', eta, optimize=True)
        # 公式：dz_{ik}=\sum_{j}^{C} da_{ij} w_{kj}
        return np.einsum('ij,kj->ik', eta, self.W.data, optimize=True)
