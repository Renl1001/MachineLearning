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
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))

        self.b = np.zeros(output_size)
        self.output = np.zeros(output_size)

        self._input = None
        self._delta = None
        self._W_grad = None
        self._b_grad = None

    def _network(self, input_sample, weight, bias):
        # weight: 4*2 4*4  input_sample: 2*107 4*107
        # bias: 4 4
        if(len(input_sample.shape) > 1)
            bias = bias[:,np.newaxis]
            bias = np.tile(bias, (1,input_sample.shape[1]))
        return np.dot(weight, input_sample) + bias

    def forward(self, input_sample):
        self._input = input_sample
        self.output = self.activator.forward(
            self._network(self._input, self.W, self.b))

    def backward(self, delta_array):
        self._delta = self.activator.backward(
            self._input) * np.dot(self.W.T, delta_array)
        self._W_grad = np.dot(delta_array.reshape(-1, 1),
                              self._input.reshape(-1, 1).T)
        self._b_grad = delta_array

    def update(self, learning_rate):
        self.W += learning_rate * self._W_grad
        self.b += learning_rate * self._b_grad


class BP(object):
    def __init__(self, layers):
        self.layers = []

        for i in range(len(layers) - 1):
            self.layers.append(Linear(
                layers[i], layers[i+1], SigmoidActivator()))

    def predict(self, sample):
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, train_sample, labels, learning_rate=0.1, epoch=5):
        for i in range(epoch):
            print('------ epoch %d -------' % (i+1))
            for j in range(len(train_sample)):
                self.predict(train_sample[j])
                self._get_gradient(labels[j])
                self._update_weight(learning_rate)

    def _get_gradient(self, label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output) * (label - self.layers[-1].output)

        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer._delta
        return delta

    def _update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)
