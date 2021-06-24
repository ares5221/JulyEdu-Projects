#!/usr/bin/env python
# _*_ coding:utf-8 _*_

## 1. 创建数据集

import pandas as pd

data = [['yes', 'no', '青年', '同意贷款'],
        ['yes', 'no', '青年', '同意贷款'],
        ['yes', 'yes', '中年', '同意贷款'],
        ['no', 'no', '中年', '不同意贷款'],
        ['no', 'no', '青年', '不同意贷款'],
        ['no', 'no', '青年', '不同意贷款'],
        ['no', 'no', '中年', '不同意贷款'],
        ['no', 'no', '中年', '不同意贷款'],
        ['no', 'yes', '中年', '同意贷款']]

# 转为 dataframe 格式
df = pd.DataFrame(data)
# 设置列名
df.columns = ['有房？', '有工作？', '年龄', '类别']

## 2. 经验熵的实现

from math import log2
from collections import Counter


def H(y):
    '''
    y: 随机变量 y 的一组观测值，例如：[1,1,0,0,0]
    '''
    # 随机变量 y 取值的概率估计值
    probs = [n / len(y) for n in Counter(y).values()]
    print(probs)
    # 经验熵：根据概率值计算信息量的数学期望
    return sum([-p * log2(p) for p in probs])


# y = [1,1,0,0,0]
# print(H(y))


## 3. 经验条件熵的实现

def cond_H(a):
    '''
    a: 根据某个特征的取值分组后的 y 的观测值，例如：
       [[1,1,1,0],
        [0,0,1,1]]
       每一行表示特征 A=a_i 对应的样本子集
    '''
    # 计算样本总数
    sample_num = sum([len(y) for y in a])
    # 返回条件概率分布的熵对特征的数学期望
    return sum([(len(y) / sample_num) * H(y) for y in a])


## 4. 特征选择函数
def feature_select(df, feats, label):
    '''
    df：训练集数据，dataframe 类型
    feats：候选特征集
    label：df 中的样本标记名，字符串类型
    '''

    # 最佳的特征与对应的信息增益比
    best_feat, gainR_max = None, -1
    # 遍历每个特征
    for feat in feats:
        # 按照特征的取值对样本进行分组，并取分组后的样本标记数组
        group = df.groupby(feat)[label].apply(lambda x: x.tolist()).tolist()
        # 计算该特征的信息增益：经验熵-经验条件熵
        gain = H(df[label].values) - cond_H(group)
        # 计算该特征的信息增益比
        gainR = gain / H(df[feat].values)

        # 更新最大信息增益比和对应的特征
        if gainR > gainR_max:
            best_feat, gainR_max = feat, gainR

    return best_feat, gainR_max


## 5. 决策树生成函数
import pickle


def creat_tree(df, feats, label):
    '''
    df：训练集数据，dataframe 类型
    feats：候选特征集，字符串列表
    label：df 中的样本标记名，字符串类型
    '''
    # 当前候选的特征列表
    feat_list = feats.copy()

    # 若当前训练数据的样本标记值只有一种
    if df[label].nunique() == 1:
        # 将数据中的任意样本标记返回，这里取第一个样本的标记值
        return df[label].values[0]
    # 若候选的特征列表为空时
    if len(feat_list) == 0:
        # 返回当前数据样本标记中的众数，各类样本标记持平时取第一个
        return df[label].mode()[0]
    # 在候选特征集中进行特征选择
    feat, gain = feature_select(df, feat_list, label)
    # 若选择的特征信息增益太小，小于阈值 0.1
    if gain < 0.1:
        # 返回当前数据样本标记中的众数
        return df[label].mode()[0]

    # 根据所选特征构建决策树，使用字典存储
    tree = {feat: {}}
    # 根据特征取值对训练样本进行分组
    g = df.groupby(feat)
    # 用过的特征要移除
    feat_list.remove(feat)
    # 遍历特征的每个取值 i
    for i in g.groups:
        # 获取分组数据，使用剩下的候选特征集创建子树
        tree[feat][i] = creat_tree(g.get_group(i), feat_list, label)

    # 存储决策树
    pickle.dump(tree, open('tree.model', 'wb'))

    return tree


# 6. 决策树分类函数
def predict(tree, feats, x):
    '''
    tree：决策树，字典结构
    feats：特征集合，字符串列表
    x：测试样本特征向量，与 feats 对应
    '''
    # 获取决策树的根结点：对应样本特征
    root = next(iter(tree))
    # 获取该特征在测试样本 x 中的索引
    i = feats.index(root)
    # 遍历根结点分裂出的每条边：对应特征取值
    for edge in tree[root]:
        # 若测试样本的特征取值=当前边代表的特征取值
        if x[i] == edge:
            # 获取当前边指向的子结点
            child = tree[root][edge]
            # 若子结点是字典结构，说明是一颗子树
            if type(child) == dict:
                # 将测试样本划分到子树中，继续预测
                return predict(child, feats, x)
            # 否则子结点就是叶子节点
            else:
                # 返回叶子节点代表的样本预测值
                return child


## 7. 在样例数据上测试

# 获取特征名列表
feats = list(df.columns[:-1])
# 获取标记名
label = df.columns[-1]
# 创建决策树（此处使用信息增益比进行特征选择）
T = creat_tree(df, feats, label)
# 计算训练集上的预测结果
preds = [predict(T, feats, x) for x in df[feats].values]
# 计算准确率
acc = sum([int(i) for i in (df[label].values == preds)]) / len(preds)
# 输出决策树和准确率
print(T, acc)
