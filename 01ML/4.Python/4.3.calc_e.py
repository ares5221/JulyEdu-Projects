#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt


def calc_e_small(x):
    # 泰勒展开求e的0.x 指数
    n = 10
    f = np.arange(1, n+1).cumprod()
    b = np.array([x]*n).cumprod()
    return np.sum(b / f) + 1


def calc_e(x):
    reverse = False
    if x < 0:   # 处理负数
        x = -x
        reverse = True
    # 计算exp(2.4)的方法 先将其变成
    ln2 = 0.69314718055994530941723212145818
    c = x / ln2
    a = int(c+0.5)
    b = x - a*ln2
    y = (2 ** a) * calc_e_small(b)
    if reverse:
        return 1/y
    return y


if __name__ == "__main__":
    # 随着横轴数据的增大 y的变化越来越明显，为了让图像好看 需要让后部分的取点更密集
    t1 = np.linspace(-2, 0, 10, endpoint=False)
    t2 = np.linspace(0, 4, 20)
    t = np.concatenate((t1, t2))
    print('t1=\n ',t1 )     # 横轴数据
    print('t2=\n ',t2 )     # 横轴数据
    print('t=\n ',t )     # 横轴数据
    y = np.empty_like(t)
    for i, x in enumerate(t):
        y[i] = calc_e(x)
        print('e^', x, ' = ', y[i], '(近似值)\t', math.exp(x), '(真实值)')
    plt.figure(facecolor='w')
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(t, y, 'r-', t, y, 'go', linewidth=2)
    plt.title('Taylor展式的应用 - 指数函数', fontsize=18)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('exp(X)', fontsize=15)
    plt.grid(True, ls=':')
    plt.show()
