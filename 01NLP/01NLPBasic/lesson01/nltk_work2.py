#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import nltk
from nltk import FreqDist
# 统计词频，讲文本转换为向量
# 做个词库先
corpus = 'this is my sentence ' \
           'this is my life ' \
           'this is the day'

# 随便tokenize一下
# 显然, 正如上文提到,
# 这里可以根据需要做任何的preprocessing:
# stopwords, lemma, stemming, etc.
tokens = nltk.word_tokenize(corpus)
print(tokens)
# 得到token好的word list
# ['this', 'is', 'my', 'sentence',
# 'this', 'is', 'my', 'life', 'this',
# 'is', 'the', 'day']
# 借用NLTK的FreqDist统计一下文字出现的频率
fdist = FreqDist(tokens)

# 它就类似于一个Dict
# 带上某个单词, 可以看到它在整个文章中出现的次数
print(fdist['is'])
# 好, 此刻, 我们可以把最常用的50个单词拿出来
standard_freq_vector = fdist.most_common(50)
size = len(standard_freq_vector)
print(standard_freq_vector)
# [('is', 3), ('this', 3), ('my', 2),
# ('the', 1), ('day', 1), ('sentence', 1),
# ('life', 1)

# Func: 按照出现频率大小, 记录下每一个单词的位置
def position_lookup(v):
    res = {}
    counter = 0
    for word in v:
        res[word[0]] = counter
        counter += 1
    return res

# 把标准的单词位置记录下来
standard_position_dict = position_lookup(standard_freq_vector)
print(standard_position_dict)
# 得到一个位置对照表
# {'is': 0, 'the': 3, 'day': 4, 'this': 1,
# 'sentence': 5, 'my': 2, 'life': 6}
# 这时, 如果我们有个新句子:
sentence = 'this is cool'
# 先新建一个跟我们的标准vector同样大小的向量
freq_vector = [0] * size
# 简单的Preprocessing
tokens = nltk.word_tokenize(sentence)
# 对于这个新句子里的每一个单词
for word in tokens:
    try:
        # 如果在我们的词库里出现过
        # 那么就在"标准位置"上+1
        freq_vector[standard_position_dict[word]] += 1
    except KeyError:
        # 如果是个新词
        # 就pass掉
        continue

print(freq_vector)
