#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    '''感知机算法的实现'''

    def __init__(self, alpha):
        ''' 初始化实例属性
        alpha: 学习率
        '''
        self.alpha = alpha

    def sign(self, x):
        '''符号函数
        x: 函数的输入值（标量）
        '''
        return 1 if x >= 0 else -1

    def predict(self, x):
        ''' 预测函数
        x: 一维数组表示的样本特征向量
        '''
        # 线性回归的预测输出
        z = sum(self.w * x) + self.b
        # 感知机的预测输出
        return self.sign(z)

    def fit(self, X_train, y_train):
        ''' 训练函数
        X_train: m x n 的 numpy 二维数组
        y_train：有 m 个元素的 numpy 一维数组
        '''
        # 初始化模型参数
        self.w = np.random.randn(X_train.shape[1])
        self.b = np.random.randn()

        # 使用随机梯度下降进行模型迭代
        while True:
            # 误分类点的个数
            num_errors = 0
            # 遍历训练集中的样本数据
            for x, y in zip(X_train, y_train):
                # 当线性回归的预测值和样本的类别取值异号时
                if y * (sum(self.w * x) + self.b) <= 0:
                    # 误分类个数加一
                    num_errors += 1
                    # 使用梯度下降法更新模型参数
                    self.w += self.alpha * y * x
                    self.b += self.alpha * y
            # 直到训练集中没有误分类点时，停止迭代
            if num_errors == 0:
                break

        return self

    def visualization_2D(self, X, y):
        '''样本在二维空间的可视化'''
        # 特征维度必须是二维，否则触发异常
        assert X.shape[1] == 2
        # 存放分离超平面的边界点
        data = []
        # 分离超平面边界点的横坐标 x1 的取值
        x1_min = X[:, 0].min() - 1
        x1_max = X[:, 0].max() + 1
        # 计算对应的纵坐标 x2 的取值
        for x1 in [x1_min, x1_max]:
            x2 = (x1 * w[0] + b) / (-w[1])
            data.append([x1, x2])
        # 将边界点转为 numpy 数组，方便取数
        data = np.array(data)
        # 绘制正样本点，标记为红色
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='r')
        # 绘制负样本点，标记为绿色
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='g')
        # 绘制分离超平面，标记为蓝色
        plt.plot(data[:, 0], data[:, 1], c='b')
        # 设置坐标轴名称
        plt.xlabel('x1')
        plt.ylabel('x2')
        # 设置纵坐标显示范围
        plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        # 显示图片
        plt.show()


from sklearn.datasets import load_iris

# 载入鸢尾花数据集（线性可分）
X, y = load_iris(return_X_y=True)
# 取前两类的数据（类别标记为0和1）
X = X[y != 2]
y = y[y != 2]

# 将类别标记中的 0 改成 -1，作为负样本
y = np.where(y == 0, -1, y)
# 设置学习率为 1，进行模型迭代
model = Perceptron(1).fit(X, y)
# 打印模型参数
print('w =', model.w, 'b =', model.b)
# 预测新数据
x_new = np.array([3, 4, 1, 5])
y_pred = model.predict(x_new)
print(x_new, y_pred)