# -*- coding: utf-8 -*-

import numpy as np

class SigmoidActivator(object):
    """
    sigmoid激活函数
    """

    def forward(self, weighted_input):
        """sigmoid激活函数

        Arguments:
            weighted_input {numpy} -- 未激活的输出结果z

        Returns:
            numpy -- sigmoid
        """
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        """sigmoid函数求导 反向传播

        Arguments:
            output {numpy} -- 输出的值

        Returns:
            numpy -- 求导后的值
        """
        return output * (1 - output)


class Linear(object):
    """
    全连接层
    """
    def __init__(self, input_size, output_size, activator):
        """
        初始化层信息，包括输入输出的大小，以及激活函数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        # 初始化参数 weight
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # 初始化参数 bias
        self.b = np.zeros(output_size)
        # 初始化该层的输出
        self.output = np.zeros(output_size)

        self.delta = None

        self._input = None
        self._W_grad = None
        self._b_grad = None

    def forward(self, input):
        """向前传播
        
        Arguments:
            input {numpy} -- 该层的输入
        """
        self._input = input
        # z = W * x + b
        Z = np.dot(self.W, input) + self.b
        self.output = self.activator.forward(Z)

    def backward(self, delta_array):
        """反向传播
        
        Arguments:
            delta_array {numpy} -- 上一层传回的误差
        """
        self.delta = self.activator.backward(
            self._input) * np.dot(self.W.T, delta_array)
        self._W_grad = np.dot(delta_array.reshape(-1, 1),
                              self._input.reshape(-1, 1).T)
        self._b_grad = delta_array

    def update(self, learning_rate):
        """更新参数
        
        Arguments:
            learning_rate {float} -- 学习率
        """
        self.W += learning_rate * self._W_grad
        self.b += learning_rate * self._b_grad


class BP(object):
    def __init__(self, layers):
        self.layers = []

        # 初始化创建每一层
        for i in range(len(layers) - 1):
            self.layers.append(Linear(
                layers[i], layers[i+1], SigmoidActivator()))

    def predict(self, test):
        """预测结果
        
        Arguments:
            test {numpy} -- 测试的输入数据
        
        Returns:
            list -- 预测的标签
        """
        preds = []
        for sample in test:
            output = sample
            for layer in self.layers:
                layer.forward(output)
                output = layer.output
            pred = 1 if output[0] > 0.5 else 0
            preds.append(pred)
        return preds

    def _forward(self, sample):
        """正向传播并预测结果
        
        Arguments:
            sample {numpy} -- 输入数据
        
        Returns:
            float -- 预测标签
        """
        output = sample
        # 循环每一层
        for layer in self.layers:
            layer.forward(output)
            output = layer.output # 将上一层的输出作为输入
        return output

    def train(self, train_sample, labels, learning_rate=0.1, epoch=200):
        """训练bp神经网络
        
        Arguments:
            train_sample {numpy} -- 训练的输入数据
            labels {numpy} -- 训练的标签
        
        Keyword Arguments:
            learning_rate {float} -- 学习率 (default: {0.1})
            epoch {int} -- 最大迭代数 (default: {200})
        """
        loss_list = []
        for i in range(epoch):
            loss = 0
            for j in range(len(train_sample)):
                # 1. 正向传播
                output = self._forward(train_sample[j])
                # 2. 计算loss
                loss += (output - labels[j])**2
                # 3. 误差反向传播
                self._backward(labels[j])
                # 4. 更新参数
                self._update_weight(learning_rate)
            
            loss_list.append(loss)
            if i > 150:
                learning_rate /= 10
            if i % 50 == 0 or i == epoch-1:
                print('------ epoch %d -------' % (i+1))
                print('loss:{}'.format(loss))
        return loss_list

    def _backward(self, label):
        """误差反向传播
        
        Arguments:
            label {list} -- 真实标签y
        """

        #计算最后一层的delta
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output) * (label - self.layers[-1].output)

        # 循环计算每一层的delta
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta

    def _update_weight(self, lr):
        """更新参数
        
        Arguments:
            lr {float} -- 学习率
        """
        for layer in self.layers:
            layer.update(lr)
