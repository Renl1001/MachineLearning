# -*- coding:utf-8 -*-
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.p0_vect = None
        self.p1_vect = None
        self.p_neg = None
        self.word_set = None

    def _create_word_set(self, data):
        """
        创建一个词库集合
        """
        word_set = set([])
        for line in data:
            word_set = word_set | set(line)
        self.word_set = list(word_set)

    def _word2vec(self, sample):
        """
        将每行的单词转化成统计出现次数的向量
        """
        sample_vector = [0] * len(self.word_set)
        for word in sample:
            if word in self.word_set:
                sample_vector[self.word_set.index(word)] += 1
        return sample_vector

    def _trainNB(self, train_samples, train_classes):
        """对数据进行训练，计算条件概率

        Arguments:
            train_samples {numpy} -- 统计单词出现次数的训练数据
            train_classes {numpy} -- 标签

        Returns:
            truple -- 正负分类的条件概率以及分类概率
        """
        numTrainDocs = len(train_samples)  # 统计样本个数
        numWords = len(train_samples[0])  # 统计特征个数，理论上是词库的长度
        p_neg = sum(train_classes) / float(numTrainDocs)  # 计算负样本出现的概率

        p0Num = np.ones(numWords)  # 初始样本个数为1，防止条件概率为0，影响结果
        p1Num = np.ones(numWords)

        p0InAll = 2.0  # 词库中只有两类，初始化为2
        p1InAll = 2.0

        #  更新正负样本数据
        for i in range(numTrainDocs):
            if train_classes[i] == 1:
                p1Num += train_samples[i]
                p1InAll += sum(train_samples[i])
            else:
                p0Num += train_samples[i]
                p0InAll += sum(train_samples[i])

        #  计算给定类别的条件下，词汇表中单词出现的概率
        #  然后取log对数，解决条件概率乘积下溢
        p0_vect = np.log(p0Num / p0InAll)  # 计算类标签为0时的其它属性发生的条件概率
        p1_vect = np.log(p1Num / p1InAll)  # log函数默认以e为底  # p(ci|w=0)
        self.p0_vect = p0_vect
        self.p1_vect = p1_vect
        self.p_neg = p_neg
        return p0_vect, p1_vect, p_neg

    def train(self, train_sample, classes):
        """ 训练 """
        self._create_word_set(train_sample)
        trainMat = []
        for postinDoc in train_sample:
            trainMat.append(self._word2vec(postinDoc))
        self._trainNB(np.array(trainMat), np.array(classes))

    def _classifyNB(self, test_vector):
        """
        使用朴素贝叶斯进行分类,返回结果为0/1
        """
        prob_y0 = sum(test_vector * self.p0_vect) + np.log(1 - self.p_neg)
        prob_y1 = sum(test_vector * self.p1_vect) + np.log(self.p_neg)  # log是以e为底
        if prob_y0 < prob_y1:
            return 1
        else:
            return 0

    def classify(self, testSample):
        """使用朴素贝叶斯进行分类,返回结果为0/1"""
        test_vector = self._word2vec(testSample)
        result = self._classifyNB(test_vector)
        print(testSample, 'classified as: ', result)
        return result
