import numpy as np

class SoftmaxRegression():

    def __init__(self, k):
        self.k = k

    def _softmax(self, x):
        """计算softmax
        
        Arguments:
            x {numpy} -- 预测的结果
        
        Returns:
            numpy -- softmax后的结果
        """
        exp = np.exp(x) # x size: m * k
        sum_exp = np.sum(np.exp(x), axis=1, keepdims=True) # size: m * 1
        softmax = exp / sum_exp

        return softmax

    def _calc_scores(self, X):
        """预测分数
        
        Arguments:
            X {numpy} -- 输入的数据X
        
        Returns:
            numpy -- 计算的分数 公式：score = X*w+b
        """
        return np.dot(X, self.weights.T) + self.bias

    def _ont_hot(self, y):
        """将label转化成是否是某种类别的矩阵
        
        Arguments:
            y {numpy} -- 数据的标签 size: (m)
        
        Returns:
            numpy -- size: m * k
        """

        one_hot = np.zeros((self.m, self.k))
        y = y.astype('int64')
        one_hot[np.arange(self.m), y.T] = 1
        return one_hot

    def fit(self, X, y, lr=0.1, maxIter=200):
        """softmax回归模型训练
        
        Arguments:
            X {numpy} -- 训练集的输入
            y {numpy} -- 训练集的标签
        
        Keyword Arguments:
            lr {float}    -- 学习率 (default: {0.01})
            maxIter {int} -- 最大迭代次数 (default: {100})
        
        Returns:
            numpy -- 权重和bias
        """
        
        self.m, n = X.shape

        self.weights = np.random.rand(self.k, n)
        self.bias = np.zeros((1, self.k))
        y_one_hot = self._ont_hot(y)

        for i in range(maxIter):
            scores = self._calc_scores(X)
            hx = self._softmax(scores)

            # 计算损失 y_one_hot size: m * k. hx size: m * k
            loss = -(1/self.m) * np.sum(y_one_hot * np.log(hx))

            # 求导
            # X size: m*n  hx size: m * k  w size: k * n
            # X.T * hx size: n * k -> w.T
            dw = (1 / self.m) * np.dot(X.T, (hx - y_one_hot))
            dw = dw.T
            db = (1 / self.m) * np.sum(hx - y_one_hot, axis=0)

            self.weights = self.weights - lr
            self.bias = self.bias - lr * db

            if i % 50 == 0:
                print('Iter:{} loss:{}'.format(i+1, loss))

        return self.weights, self.bias
    
    def predict(self, X):
        """使用学习得到模型进行分类
        
        Arguments:
            X {numpy} -- 测试集的输入值
        
        Returns:
            numpy -- 预测结果
        """
        scores = self._calc_scores(X)
        hx = self._softmax(scores) # softmax
        # hx size: m*k
        pred = np.argmax(hx, axis=1)[:,np.newaxis]
        return pred

if __name__ == "__main__":
    x = np.asarray([[1,2,3,4],[1,2,3,4]])
    print(x)
    sr = SoftmaxRegression(3)
    sr._softmax(x)
