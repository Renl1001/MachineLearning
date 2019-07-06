# -*- coding: utf-8 -*-

import numpy as np
import operator

class KNN():

    def __init__(self, k=3):
        self._k = k
        
    
    def _eclud_dist(self, test, train_X):
        """计算欧式距离并排序
        
        Arguments:
            test {list} -- 测试点的X
            train_X {list} -- 训练集
        
        Returns:
            numpy -- 测试点与每个点之间的距离进行排序后的下标
        """
        
        m = train_X.shape[0]   # 得到训练集的数量
        diffMat = np.tile(test, (m,1)) - train_X # 计算测试点的每个维度与训练数据的差值
        sqDiffMat = diffMat**2  # 每个元素求平方
        sqDistances = sqDiffMat.sum(axis = 1)  # 计算得到差值的平方和
        distances = sqDistances ** 0.5 # 开根得到欧式距离
        return distances.argsort()  #按距离的从小到达排列的下标值
        
    
    def classify_one(self, sample, train_X, train_y):
        """对一个样本进行分类
        
        Arguments:
            sample {list} -- 要分类的点的坐标
            train_X {list} -- 训练集的X
            train_y {list} -- 训练集的y
        
        Returns:
            int -- sample的标签y
        """
        sortedDistances = self._eclud_dist(sample, train_X)
        classCount = {}
        max_y = -1
        max_num = 0
        for i in range(self._k):
            oneVote = train_y[sortedDistances[i]] #获取最近的第i个点的类别
            classCount[oneVote] = classCount.get(oneVote, 0) + 1
            if  classCount[oneVote] > max_num:
                max_num = classCount[oneVote]
                max_y = oneVote

        return max_y
    
    def classify(self, test_X, train_X, train_y):
        """对测试集test_X进行分类
        
        Arguments:
            test_X {list} -- 测试集的X
            train_X {list} -- 训练集的X
            train_y {list} -- 训练集的y
        
        Returns:
            list -- 测试集的标签
        """
        labels = []

        for i in range(len(test_X)):
            sample = test_X[i]
            label = self.classify_one(sample, train_X, train_y)
            labels.append(label)
        return labels
        
        
if __name__=="__main__":
    train_X = [[1, 2, 0, 1, 0],
               [0, 1, 1, 0, 1],
               [1, 0, 0, 0, 1],
               [2, 1, 1, 0, 1],
               [1, 1, 0, 1, 1]]
    train_y = [1, 1, 0, 0, 0]
    clf = KNN(k = 3)
    sample = [[1,2,0,1,0],[1,2,0,1,1]]
    result = clf.classify(sample, train_X, train_y)