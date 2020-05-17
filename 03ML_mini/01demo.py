#!/usr/bin/env python
# _*_ coding:utf-8 _*_

# 1. 导入相关的工具包

# 使用 sklearn 内置的波士顿房价数据集，load_boston 是加载数据集的函数
from sklearn.datasets import load_boston

# 使用sklearn 中的 train_test_split 划分数据集
from sklearn.model_selection import train_test_split

# 使用 sklearn 中的线性回归模型进行预测
from sklearn.linear_model import LinearRegression
# 使用 matplotlib 中的 pyplot 进行可视化
import matplotlib.pyplot as plt
# 2. 加载数据集

# ToDo：加载波士顿房价数据集，返回特征X和标签y
X, y =  load_boston(return_X_y=True)

# 只取第6列特征（方便可视化）：住宅平均房间数
X = X[:,5:6]

# ToDo：划分为训练集和测试集，测试集取20%
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, random_state=2020)

# 3. 模型训练和预测

# 创建线性回归对象
regr = LinearRegression()

# ToDo：使用训练集训练模型
regr.fit(X_train, y_train)

# ToDo：在测试集上进行预测
y_pred =regr.predict(X_test)

# 4. 打印前3个预测值和真实值

print('y_pred:',y_pred[:3])
print('y_test:',y_test[:3])

# 5. ToDo：打印斜率和截距
# 画测试数据散点图
plt.scatter(X_test, y_test,  color='blue')

# 画线性回归模型对测试数据的拟合曲线
plt.plot(X_test, y_pred, color='red')

# 显示绘图结果
plt.show()
print('斜率：{}, 截距：{}'.format(regr.coef_,regr.intercept_))