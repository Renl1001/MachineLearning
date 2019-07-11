# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
from softmax_regression import SoftmaxRegression
import matplotlib.pyplot as plt

# 加载数据集，DataFrame格式，最后将返回为一个matrix格式
def loadDataset(infile):
    df = pd.read_csv(infile, sep='\t', header=0, dtype=str, na_filter=False)
    data = np.array(df).astype(np.float)
    return data

# 绘制图形
def show_data(data, label, title, weights=None):
    plt.cla()
    plt.scatter(data[:, 0], data[:, 1], c=label)
    if weights is not None:
        weights = weights.getA()
        print(weights[0],weights[1])
        x = np.arange(0, 10, 0.1)
        y =  - weights[0] * x / weights[1]
        plt.plot(x, y)

    plt.title(title)
    plt.axis([-10,10,-10,10])
    outname = "./save/{}.png".format(title)
    plt.savefig(outname)
    plt.show()

if __name__=='__main__':
    train = loadDataset('train.txt')
    test = loadDataset('test.txt')
    train_X = train[:, :2]
    train_y = train[:, 2]
    test_X = test[:,:2]

    clf = SoftmaxRegression(4)
    weights = clf.fit(train_X, train_y)
    show_data(train[:,:2], train[:, 2], 'train')
    pre_y = clf.predict(test_X)
    show_data(test_X, pre_y[:,0], 'test')
    