import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pca import PCA


def loaddata(datafile):
    return np.array(pd.read_csv(datafile, sep="\t",
                                header=-1)).astype(np.float)


def plotBestFit(data1):
    dataArr1 = np.array(data1)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    for i in range(m):
        axis_x1.append(dataArr1[i, 0])
        axis_y1.append(dataArr1[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig("save/output.png")
    plt.show()


# 根据数据集data.txt
def main():
    datafile = "data.txt"
    data = loaddata(datafile)
    k = 2
    pca = PCA(k)
    return pca.fit_transform(data)


if __name__ == "__main__":
    new_data = main()
    plotBestFit(new_data)
