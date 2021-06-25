#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.datasets import load_iris


class Kmeans:

    def __init__(self, k):
        ''' 初始化实例属性

        Parameters
        ----------
        k: 聚类中心的个数
        '''
        self.k = k
        # 存放 k 个聚类中心的数组
        self.center = np.array([])
        # 存放聚类样本子集的字典
        self.center_data = dict()

    def distance(self, u, v):
        ''' 样本距离的定义

        Parameters
        ----------
        u,v: 一维 numpy 数组表示的特征向量
        '''
        # 返回欧氏距离的平方
        return  sum((u-v)**2)

    def center_init(self, data):
        ''' 初始化聚类中心 kmeans

        从 data 中随机选择 k 个样本特征向量，作为初始聚类中心

        Parameters
        ----------
        data: 存放样本特征向量的 numpy array
        '''
        # 随机选择 k 个样本特征向量的索引
        index = np.random.choice(a=range(len(data)), size=self.k)
        # 根据索引获得 k 个聚类中心
        return data[index]

    def center_init2(self,data, k):
        '''
        kmeans++
        data: numpy array,样本特征向量集合
        k: int,聚类中心的个数
        '''
        # 用来保存聚类中心
        center = []
        # 随机选择一个样本特征向量的索引
        index = np.random.choice(range(len(data)))
        # 根据索引获得第一个聚类中心
        center.append(data[index])
        # 获取剩余 k-1 个聚类中心
        for _ in range(k - 1):
            # 样本被选为聚类中心的概率
            prob = []
            # 遍历每个样本数据
            for x in data:
                # 根据步骤 2 的公式计算样本与最近类中心的距离
                d = min([self.distance(x, c) for c in center])
                prob.append(d)
            # 计算每个样本的被选概率
            prob = np.array(prob) / sum(prob)
            # 根据被选中的概率随机选择一个样本特征向量的索引
            i = np.random.choice(a=range(len(data)), p=prob)
            # 根据索引添加一个聚类中心
            center.append(data[i])
        # 返回 k 个初始化的聚类中心
        return np.array(center)

    def center_init3(self,data, k):
        '''
        kmeans++2
        data: numpy array,样本特征向量集合
        k: int,聚类中心的个数
        '''
        # 用来保存聚类中心
        center = []
        # 随机选择一个样本特征向量的索引
        index = np.random.choice(range(len(data)))
        # 根据索引获得第一个聚类中心
        center.append(data[index])
        # 获取剩余 k-1 个聚类中心
        for _ in range(k - 1):
            # 存放样本与最近聚类中心的距离
            arr = []
            # 遍历每个样本数据
            for x in data:
                # 根据步骤 2 的公式计算样本与最近类中心的距离
                d = min([self.distance(x, c) for c in center])
                arr.append(d)
            # 选择 d 值最大的样本索引
            i = np.argmax(arr)
            # 根据索引添加一个聚类中心
            center.append(data[i])
        # 返回 k 个初始化的聚类中心
        return np.array(center)


    def predict(self, x):
        ''' 获取样本 x 的聚类索引

        Parameters
        ----------
        x: 一维数组表示的样本特征向量
        '''
        # 输入的样本 x 必须为一维数组，否则触发异常
        assert np.array(x).ndim == 1
        # 返回距离样本 x 最近的聚类中心索引
        return  np.argmin([self.distance(x,c) for c in self.center])

    def cluster(self, data):
        ''' 获取聚类中心对应的样本子集

        Parameters
        ----------
        data: 存放样本特征向量的 numpy array
        '''
        # 存放各类样本子集的字典：key = 聚类中心索引; value = data 子集
        center_data = dict()
        # 遍历样本特征向量
        for x in data:
            # 获取样本 x 的聚类索引
            i = self.predict(x)
            # 将 x 添加到聚类索引对应的样本子集中
            center_data.setdefault(i, [])
            center_data[i].append(x)
        # 返回各聚类中心对应的样本子集
        return center_data

    def visualization_2D(self):
        ''' 二维空间可视化'''
        # 设置画布大小
        plt.figure(figsize=(8, 6))
        # 设置可选颜色：红、绿、蓝
        color = ['r', 'g', 'b']
        # 遍历每个聚类中心的索引
        for i in self.center_data:
            # 根据索引，取该聚类中心对应的样本第一列特征
            x = np.array(self.center_data[i])[:, 0]
            # 根据索引，取该聚类中心对应的样本第二列特征
            y = np.array(self.center_data[i])[:, 1]
            # 绘制当前聚类的样本散点图，s 控制大小，c 控制颜色
            plt.scatter(x, y, s=30, c=color[i])
        # 遍历每个聚类中心 c 和对应的索引 i
        for i, c in enumerate(self.center):
            # 在二维平面上绘制聚类中心，alpha 控制透明度
            plt.scatter(x=c[0], y=c[1], s=200, c=color[i], alpha=0.2)
        # 显示图片
        plt.show()

    def visualization_3D(self):
        ''' 三维空间可视化'''
        # 取第一个聚类的样本子集，若样本的特征维数小于 3
        if np.array(self.center_data[0]).shape[1] < 3:
            # 则返回空值，放弃三维可视化
            return None
        # 设置画布大小
        plt.figure(figsize=(10, 8))
        # 设置可选颜色：红、绿、蓝
        color = ['r', 'g', 'b']
        # 创建三维空间直角坐标系
        ax = plt.axes(projection='3d')
        # 遍历每个聚类中心的索引
        for i in self.center_data:
            # 根据索引，取该聚类中心对应的样本第一列特征
            x = np.array(self.center_data[i])[:, 0]
            # 根据索引，取该聚类中心对应的样本第二列特征
            y = np.array(self.center_data[i])[:, 1]
            # 根据索引，取该聚类中心对应的样本第三列特征
            z = np.array(self.center_data[i])[:, 2]
            # 绘制当前聚类的样本散点图，s 控制大小，c 控制颜色
            ax.scatter3D(x, y, z, s=30, c=color[i])
        # 遍历每个聚类中心 c 和对应的索引 i
        for i, c in enumerate(self.center):
            # 在三维空间中绘制当前聚类中心，alpha 控制透明度
            ax.scatter3D(xs=c[0], ys=c[1], zs=c[2], s=200, c=color[i], alpha=0.2)
        # 显示图片
        plt.show()

    def fit(self, data, visualization=None):
        ''' 算法迭代过程

        Parameters
        ----------
        data: 存放样本特征向量的 numpy array
        visualization： 可视化选项，默认为 None，不对聚类过程做可视化

        Returns
        -------
        self
        '''
        # 初始化聚类中心
        self.center = self.center_init(data)
        # 循环迭代
        while True:
            # 获取聚类中心对应的样本子集
            self.center_data = self.cluster(data)
            # 当前聚类中心和样本特征向量在二维空间的可视化
            if visualization == '2d':
                self.visualization_2D()
            # 当前聚类中心和样本特征向量在三维空间的可视化
            if visualization == '3d':
                self.visualization_3D()
            # 保存上一次迭代的聚类中心
            old_center = self.center.copy()
            # 遍历各个聚类中心的索引
            for i in self.center_data:
                # 更新每一个聚类中心：样本子集特征向量的均值
                self.center[i] = np.mean(self.center_data[i],axis=0)
            # 循环迭代的停止条件：最近两次迭代，聚类中心的位置不再变化
            if sum([self.distance(self.center[i], old_center[i]) for i in range(self.k)]) == 0:
                break

        return self


# 加载鸢尾花数据集的特征向量
data = load_iris().data
# Kmeans 聚类
model = Kmeans(3).fit(data)
# 打印聚类中心
print(model.center)