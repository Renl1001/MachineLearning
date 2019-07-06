# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
from KNN import KNN
import matplotlib.pyplot as plt

# 加载数据集，DataFrame格式，最后将返回为一个matrix格式
def loadDataset(infile):
    df = pd.read_csv(infile, sep='\t', header=0, dtype=str, na_filter=False)
    data = np.array(df).astype(np.float)
    return data

def show_data(data):
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
    
    plt.title("all point")
    plt.axis([0,10,0,10])
    outname = "./save/data.png"
    plt.savefig(outname)
    plt.show()

# 绘制图形
def show_result(data, k, test_X, labels):
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2]) 
    plt.scatter(test_X[:, 0], test_X[:, 1], c=labels,marker='x')
    
    plt.title("result")
    plt.axis([0,10,0,10])
    outname = "./save/result" + str(k) + ".png"
    plt.savefig(outname)
    plt.show()

if __name__=='__main__':
    data = loadDataset('data.txt')
    show_data(data)
    k = 3
    clf = KNN(k)
    test_X = [2,3]
    label = clf.classify_one(test_X, data[:,:2], data[:,2])
    test_X = np.array([test_X])
    label = np.array([label])
    show_result(data, 'one', test_X, label)

    test_X = []
    for i in range(10):
        x = random.random()*10
        y = random.random()*10
        test_X.append([x,y])
    
    labels = clf.classify(test_X, data[:,:2], data[:,2])
    test_X = np.array(test_X)
    labels = np.array(labels)
    show_result(data, 'all', test_X, labels)
    