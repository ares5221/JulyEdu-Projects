#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import nltk
# 运行环境为win10 pycharm python3.8
# pycharm中采用setting直接安装nltk，然后执行下面的语句
# 下载语料
# nltk.download()
# 弹出下载框，自动下载

from nltk.corpus import brown
print(brown.categories())
# 验证数据是否安装成功，若报错根据报错修改，存储位置根据推荐的方式保存位置E:\nltk_data，可用国内镜像源替换下载地址
# nltk.download('punkt')

print(len(brown.words()))
print(len(brown.sents()))

print('test Tokenize')
import nltk
sentence = "hello, world"
tokens = nltk.word_tokenize(sentence)
print(tokens)

import jieba
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print ("Full Mode:", "/ ".join(seg_list) ) # 全模式
seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print ("Default Mode:", "/ ".join(seg_list))  # 精确模式
seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print (", ".join(seg_list))
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
# 搜索引擎模式
print (", ".join(seg_list))

from nltk.tokenize import word_tokenize
tweet = 'RT @angelababy: love you baby! :D http://ah.love #168cm'
print(word_tokenize(tweet))





