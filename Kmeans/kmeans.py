import numpy as np

class KMeans():
    
    def __init__(self, k=3):
        
        self._k = k
        self._labels = None
        self._mdist = None
        self.step = None
        
    def _eclud_dist(self, p1, p2):
        """计算两点之间的欧拉距离
        
        Arguments:
            p1 {numpy} -- p1的坐标
            p2 {numpy} -- p2的坐标
        
        Returns:
            float -- p1和p2的欧拉距离
        """
        dist = np.sqrt(np.sum((p1-p2)**2)) 
        return dist

    def _rand_Centroid(self, data, k):
        """随机初始化质心
        
        Arguments:
            data {numpy} -- 所有点的坐标
            k {int} -- 分类数
        
        Returns:
            numpy -- 质心数组 k * m
        """
        m = data.shape[1]   # 获取特征的维数
        centroids = np.empty((k,m)) # 生成 k * m 的矩阵，用于存储质心
        for i in range(m):
            min_data = min(data[:, i])  # 计算第 i 维的最小值
            max_data = max(data[:, i])  # 计算第 i 维的最大值
            range_data = min_data + (max_data - min_data) * np.random.rand(k)   # 随机生成 k 个范围在[min_data, max_data]之间的数
            centroids[:, i] = range_data
        return centroids
    
    def fit(self, data):
        """k均值聚类的实现
        
        Arguments:
            data {numpy} -- 点的数据
        """
                
        n = data.shape[0]  #获取样本的个数
        
        data_index = np.zeros(n) # 记录每个点对应的质心下标
        data_min = np.zeros(n) # 记录每个点到质心的最短距离

        for i in range(n):
            data_min[i] = np.inf

        self._centroids = self._rand_Centroid(data, self._k)
            
        for step in range(500): 
            self.step = step+1
            flag = False # 用来记录是否有点改变了质心
            for i in range(n):   # 循环遍历n个点
                p1 = data[i,:]   # 点的坐标
                minDist = np.inf # 初始化点到质心的最小距离和下标
                minIndex = -1
                
                for j in range(self._k): # 遍历 k 个质心
                    p2 = self._centroids[j,:] # 质心坐标
                    dist = self._eclud_dist(p2, p1) # 计算距离
                    if dist < minDist: # 更新最短距离
                        minDist = dist
                        minIndex = j

                if data_index[i] != minIndex:
                    flag = True
                    data_index[i] = minIndex
                    data_min[i] = minDist**2

            if not flag: # 当所有点都没有更新质点的时候结束迭代
                print('迭代次数：', step)
                break
                
            '''
            更新质心
            将质心中所有点的坐标的平均值作为新质心
            '''
            for i in range(self._k): 
                index_all = data_index #取出样本所属簇的索引值
                value = np.nonzero(index_all==i) #取出所有属于第i个簇的索引值
                ptsInClust = data[value[0]]    #取出属于第i个簇的所有样本点
                self._centroids[i,:] = np.mean(ptsInClust, axis=0) #计算均值
        
        self._labels = data_index
        self._mdist = sum(data_min)
    
    def predict(self, X):
        """根据训练结果，来预测新的数据的分类
        
        Arguments:
            X {numpy} -- 要预测的数据
        
        Returns:
            numpy -- 预测结果
        """
        n = X.shape[0] # 样本数量
        preds = np.empty((n,))
        for i in range(n):
            minDist = np.inf # 记录最短距离
            for j in range(self._k):
                distJI = self._eclud_dist(self._centroids[j,:], X[i,:])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds

if __name__ == "__main__":
    '''
    测试欧拉距离函数
    '''
    kmeans = KMeans(4)
    p1 = np.array([0,0])
    p2 = np.array([1,1])
    print(kmeans._eclud_dist(p1, p2))
    p1 = np.array([0,0])
    p2 = np.array([3,4])
    print(kmeans._eclud_dist(p1, p2))