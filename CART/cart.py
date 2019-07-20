class Tree:
    def __init__(self,
                 value=None,
                 trueBranch=None,
                 falseBranch=None,
                 results=None,
                 col=-1,
                 summary=None,
                 data=None):
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results
        self.col = col
        self.summary = summary


class CART:
    def __init__(self, miniGain=0.4):
        self.decisionTree = None
        self.miniGain = miniGain

    def _cal_labels_count(self, datas):
        """统计输入数据中每类标签的数量

        Arguments:
            datas {numpy} -- 一个二维数组，最后一列为标签

        Returns:
            dictionary -- 记录每类标签的一个字典
        """

        labels_count = {}
        for data in datas:  # 统计标签的数量
            # data[-1] means dataType
            label = data[-1]
            if label not in labels_count:
                labels_count[label] = 1
            else:
                labels_count[data[-1]] += 1
        return labels_count

    def _gini(self, datas):
        """计算gini值

        Arguments:
            datas {numpy} -- 二维数组，最后一列表示标签

        Returns:
            float -- gini值
        """

        length = len(datas)
        labels_count = self._cal_labels_count(datas)

        # 计算gini值
        gini = 0.0
        for i in labels_count:
            gini += (labels_count[i] / length)**2
        gini = 1 - gini

        return gini

    def split_datas(self, datas, value, column):
        """将数据通过指定的列的值分割成两个数据

        Arguments:
            datas {numpy} -- 待分割的数据集
            value {int or float or string} -- 分割参考值
            column {int} -- 分割时使用的列

        Returns:
            truple -- 分割后的两个数据集
        """
        data1 = []
        data2 = []
        if (isinstance(value, int) or isinstance(value, float)):  # 连续型数据
            for row in datas:
                if (row[column] >= value):
                    data1.append(row)
                else:
                    data2.append(row)
        else:  # 标签型数据
            for row in datas:
                if row[column] == value:
                    data1.append(row)
                else:
                    data2.append(row)

        return (data1, data2)

    def buildDecisionTree(self, data):
        '''
        建立决策树
        '''
        self.decisionTree = self._build(data)

    def _build(self, datas):
        """递归建立决策树

        Arguments:
            datas {numpy} -- 训练的数据集

        Returns:
            Tree -- 树的节点（叶子和非叶子）
        """

        current_gain = self._gini(datas)
        column_length = len(datas[0])
        datas_length = len(datas)
        best_gain = 0.0
        best_value = None
        best_set = None

        # 找到最大的gain以及决定它的列和值
        for col in range(column_length - 1):
            col_value_set = set([x[col] for x in datas])
            for value in col_value_set:
                data1, data2 = self.split_datas(datas, value, col)
                p = len(data1) / datas_length
                gain = current_gain - p * self._gini(data1) - (
                    1 - p) * self._gini(data2)
                if gain > best_gain:
                    best_gain = gain
                    best_value = (col, value)
                    best_set = (data1, data2)

        dcY = {
            'impurity': '%.3f' % current_gain,
            'samples': '%d' % datas_length
        }

        # 通过miniGain进行剪枝
        if best_gain > self.miniGain:
            trueBranch = self._build(best_set[0])
            falseBranch = self._build(best_set[1])

            # 非叶子节点需要保存列号，分割的值以及左右子树
            return Tree(col=best_value[0],
                        value=best_value[1],
                        trueBranch=trueBranch,
                        falseBranch=falseBranch,
                        summary=dcY)
        else:
            # 叶子节点保存值
            return Tree(results=self._cal_labels_count(datas),
                        summary=dcY)

    def classify(self, data):
        '''
        利用决策树进行分类
        '''
        return self._classify(data, self.decisionTree)

    def _classify(self, data, tree):
        """递归查找决策树，查找分类

        Arguments:
            data {numpy} -- 测试数据
            tree {Tree} -- 树的节点

        Returns:
            string -- 类别
        """

        # 叶子节点直接返回
        if tree.results is not None:
            return tree.results

        branch = None
        v = data[tree.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        else:
            if v == tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        return self._classify(data, branch)
