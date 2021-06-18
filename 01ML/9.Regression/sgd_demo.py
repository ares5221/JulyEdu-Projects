#!/usr/bin/env python
# _*_ coding:utf-8 _*_

# 1. 数据加载和预处理

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 获取波士顿房价数据集
boston = load_boston()
# 获取数据集特征
X = boston.data
# 获取数据集标记
y = boston.target
# 特征归一化到 [0,1] 范围内：提升模型收敛速度
X = MinMaxScaler().fit_transform(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)

# 2. 线性回归算法实现

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    '''线性回归算法实现'''

    def __init__(self, alpha=0.1, epoch=5000, fit_bias=True):
        '''
        alpha: 学习率，控制参数更新的幅度
        epoch: 在整个训练集上训练迭代（参数更新）的次数
        fit_bias: 是否训练偏置项参数
        '''
        self.alpha = alpha
        self.epoch = epoch
        # cost_record 记录每一次迭代的经验风险
        self.cost_record = []
        self.fit_bias = fit_bias

    # 预测函数
    def predict(self, X_test):
        '''
        X_test: m x n 的 numpy 二维数组
        '''
        # 模型有偏置项参数时：为每个测试样本增加特征 x_0 = 1
        if self.fit_bias:
            x_0 = np.ones(X_test.shape[0])
            X_test = np.column_stack((x_0, X_test))

        # 根据公式返回结果
        return np.dot(X_test, self.w)

    # 模型训练：使用梯度下降法更新参数
    def fit(self, X_train, y_train):
        '''
        X_train: m x n 的 numpy 二维数组
        y_train：有 m 个元素的 numpy 一维数组
        '''
        # 训练偏置项参数时：为每个训练样本增加特征 x_0 = 1
        if self.fit_bias:
            x_0 = np.ones(X_train.shape[0])
            X_train = np.column_stack((x_0, X_train))

        # 训练样本数量
        m = X_train.shape[0]
        # 样本特征维数
        n = X_train.shape[1]
        # 初始模型参数
        self.w = np.ones(n)

        # 模型参数迭代
        for i in range(self.epoch):
            # 计算训练样本预测值
            y_pred = np.dot(X_train, self.w)
            # 计算训练集经验风险
            cost = np.dot(y_pred - y_train, y_pred - y_train) / (2 * m)
            # 记录训练集经验风险
            self.cost_record.append(cost)
            # 参数更新GD
            self.w -= self.alpha/m * np.dot(y_pred - y_train, X_train)

        # 保存模型
        self.save_model()

    # 显示经验风险的收敛趋势图
    def polt_cost(self):
        plt.plot(np.arange(self.epoch), self.cost_record)
        plt.xlabel("epoch")
        plt.ylabel("cost")
        plt.show()

    # 保存模型参数
    def save_model(self):
        np.savetxt("model.txt", self.w)

    # 加载模型参数
    def load_model(self):
        self.w =  np.loadtxt('model.txt')


# 3. 模型的训练和预测

# 实例化一个对象
model = LinearRegression()
# 在训练集上训练
model.fit(X_train, y_train)
# 在测试集上预测
y_pred = model.predict(X_test)

# 4. ToDo：打印模型参数
print('偏置参数：', model.w[0])
print('特征权重：', model.w)

# 5. 打印测试集前5个样本的预测结果
print('预测结果：', y_pred[:5])

model.polt_cost()

# # 实例化一个对象
# model = LinearRegression()
# # 加载训练好的模型参数
# model.load_model()
# # 在测试集上预测
# y_pred = model.predict(X_test)