#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 导入乳腺癌数据集
x,y = load_breast_cancer(return_X_y=True)
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
# 实例化对象
rf = RandomForestClassifier(n_estimators=8, criterion='gini',max_depth=None, n_jobs=-1)
# 根据训练数据建立随机森林
rf = rf.fit(x_train, y_train)
# 模型预测
y_pred = rf.predict(x_test)
# 模型评估：计算准确率
accuracy = rf.score(x_test, y_test)
print(accuracy)
# 测试样本分别归属于每颗决策树的叶子结点索引
x_trans = rf.apply(x_test)
print(x_trans)