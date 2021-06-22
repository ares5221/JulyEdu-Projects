#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 获取乳腺癌数据集
cancer = load_breast_cancer()
# 获取数据集特征
X = cancer.data
# 获取数据集标记
y = cancer.target
# 特征归一化到 [0,1] 范围内：提升模型收敛速度
X = MinMaxScaler().fit_transform(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2020)

class LogisticRegression:
    '''逻辑回归算法实现'''

    def __init__(self, alpha=0.1, epoch=5000, fit_bias=True, threshold=0.5):
        '''
        alpha: 学习率，控制参数更新的幅度
        epoch: 在整个训练集上训练迭代（参数更新）的次数
        fit_bias: 是否训练偏置项参数
        threshold：判定为正类的概率阈值
        '''
        self.alpha = alpha
        self.epoch = epoch
        # cost_record 记录每一次迭代后的经验风险
        self.cost_record = []
        self.fit_bias = fit_bias
        self.threshold = threshold

    # 概率预测函数
    def predict_proba(self, X_test):
        '''
        X_test: m x n 的 numpy 二维数组
        '''
        # 模型有偏置项参数时：为每个测试样本增加特征 x_0 = 1
        if self.fit_bias:
            x_0 = np.ones(X_test.shape[0])
            X_test = np.column_stack((x_0, X_test))

        # 根据预测公式返回结果
        z = np.dot(X_test, self.w)
        return 1 / (1 + np.exp(-z))

    # 类别预测函数
    def predict(self, X_test):
        '''
        X_test: m x n 的 numpy 二维数组
        '''
        probs = self.predict_proba(X_test)
        results = map(lambda x: int(x > self.threshold), probs)
        return np.array(list(results))

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
            z = np.dot(X_train, self.w)
            y_pred = 1 / (1 + np.exp(-z))
            # 计算训练集经验风险
            cost = -(np.dot(y_train, np.log(y_pred)) +
                     np.dot(np.ones(m) - y_train, np.log(np.ones(m) - y_pred))) / m
            # 记录训练集经验风险
            self.cost_record.append(cost)
            # 参数更新
            self.w += self.alpha / m * np.dot(y_train - y_pred, X_train)

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
        self.w = np.loadtxt('model.txt')


from sklearn import linear_model

# 实例化一个对象
model_1 = LogisticRegression(epoch=60000)
model_2 = linear_model.LogisticRegression()
# 在训练集上训练
model_1.fit(X_train,y_train)
model_2.fit(X_train,y_train)
# 在测试集上预测
y_pred_1 = model_1.predict(X_test)
y_pred_2 = model_2.predict(X_test)
model_1.polt_cost()
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

metrics = dict()

acc_1 = accuracy_score(y_test,y_pred_1)
acc_2 = accuracy_score(y_test,y_pred_2)
metrics['准确率'] = [acc_1,acc_2]

pre_1 = precision_score(y_test,y_pred_1)
pre_2 = precision_score(y_test,y_pred_2)
metrics['精确率'] = [pre_1,pre_2]

rec_1 = recall_score(y_test,y_pred_1)
rec_2 = recall_score(y_test,y_pred_2)
metrics['召回率'] = [rec_1,rec_2]

f1_1 = f1_score(y_test,y_pred_1)
f1_2 = f1_score(y_test,y_pred_2)
metrics['F1值'] = [f1_1,f1_2]

auc_1 = roc_auc_score(y_test, model_1.predict_proba(X_test))
auc_2 = roc_auc_score(y_test, model_2.predict_proba(X_test)[:,1])
metrics['AUC'] = [auc_1,auc_2]

df = pd.DataFrame(metrics,index=['model_1','model_2'])
print(df)