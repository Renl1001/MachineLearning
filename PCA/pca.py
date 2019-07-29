# -*- coding: utf-8 -*-

import numpy as np


class PCA:
    def __init__(self, k):
        self.k = k

    def zero_mean(self, data):
        """零均值化

        Arguments:
            data {numpy} -- 传入数据，m行n列表示m条数据n维特征

        Returns:
            numpy, numpy -- 零均值化后的数据和均值
        """
        # print('data.shape:', data.shape)
        mean_data = np.mean(data, axis=0)  # 对列求均值
        # print('mean.shape:', mean_data.shape)
        mean_data = np.tile(mean_data, (data.shape[0], 1))  # 对源数据进行复制扩充到m行
        data = data - mean_data
        return data, mean_data

    def fit_transform(self, data):
        """计算pca

        Arguments:
            data {numpy} -- 原始待降维数据 m * n

        Returns:
            numpy -- 降维后的数据 m * k
        """
        data, mean_data = self.zero_mean(data)
        m, n = np.shape(data)  # m个数据n个特征
        covX = np.cov(data.T)  # 计算协方差矩阵
        feat_value, feat_vec = np.linalg.eig(covX)  # 求解协方差矩阵的特征值和特征向量
        # print('feat_value:', feat_value)
        # print('feat_vec', feat_vec)

        index = np.argsort(-feat_value)  # 按照特征值进行从大到小排序返回下标
        k_vector = np.matrix(feat_vec.T[index[:self.k]]).T  # 取前k项
        new_data = data * k_vector
        # print(new_data)
        return new_data
