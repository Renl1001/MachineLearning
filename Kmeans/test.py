# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from kmeans import KMeans
import matplotlib.pyplot as plt

# 加载数据集，DataFrame格式，最后将返回为一个matrix格式
def loadDataset(infile):
    df = pd.read_csv(infile, sep='\t', header=0, dtype=str, na_filter=False)
    data = np.array(df).astype(np.float)
    return data

def show_data(data):
    for i in range(data.shape[0]):
        x0 = data[:, 0]
        x1 = data[:, 1]
        plt.scatter(x0[i], x1[i])
    
    plt.title("all point")
    plt.axis([0,10,0,10])
    outname = "./save/data.png"
    plt.savefig(outname)
    plt.show()

# 绘制图形
def show_result(data, k, centroids, labels, step):
    colors = ['b','g','r','k','c','m','y']
    for i in range(k):
        index = np.nonzero(labels==i)[0]
        x0 = data[index, 0]
        x1 = data[index, 1]
        for j in range(len(x0)):
            plt.scatter(x0[j], x1[j], color=colors[i])
           
        plt.scatter(centroids[i,0],centroids[i,1],marker='x',color=colors[i],\
                    linewidths=7)
    
    plt.title("step={}".format(step))
    plt.axis([0,10,0,10])
    outname = "./save/result" + str(k) + ".png"
    plt.savefig(outname)
    plt.show()

if __name__=='__main__':
    data = loadDataset('data.txt')
    show_data(data)
    k = 4
    clf = KMeans(k)
    clf.fit(data)
    show_result(data, k, clf._centroids, clf._labels, clf.step)
    
    