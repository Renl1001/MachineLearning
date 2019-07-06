import numpy as np

class LogisticRegression():
    
    def __init__(self):
        self._alpha = None
        self._w = None

    def _sigmoid(self, x):
        """sigmoid函数
        
        Arguments:
            x {numpy} -- 输入的x
        
        Returns:
            numpy -- sigmoid(x)
        """
        return 1.0/(1 + np.exp(-x))

    def fit(self, train_X, train_y, alpha=0.01, maxIter=200):
        """逻辑回归训练
        
        Arguments:
            train_X {numpy} -- 训练集的输入
            train_y {numpy} -- 训练集的标签
        
        Keyword Arguments:
            alpha {float} -- 学习率 (default: {0.01})
            maxIter {int} -- 最大迭代次数 (default: {100})
        
        Returns:
            numpy -- 权重
        """
        dataMat = np.mat(train_X)           # size: m*n
        labelMat = np.mat(train_y).T        # size: m*1
        n = dataMat.shape[1]
        weights = np.random.random((n, 1)) 
        for _ in range(maxIter):
            hx = self._sigmoid(dataMat * weights)  # 1. 计算预测函数h(x)
            J = labelMat - hx                      # 2. 计算损失函数J(w)
            weights = weights + alpha * dataMat.transpose() * J # 3. 根据误差修改回归权重参数
        self._w = weights
        return weights

    #使用学习得到的参数进行分类
    def predict(self, test_X):
        """使用学习得到的参数进行分类
        
        Arguments:
            test_X {numpy} -- 测试集的输入值
        
        Returns:
            numpy -- 预测结果
        """
        dataMat = np.mat(test_X)
        hx = self._sigmoid(dataMat*self._w) 
        hx = hx.getA()
        pre_y = hx > 0.5 # 概率大于0.5则标签为1
        return pre_y