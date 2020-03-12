#!/usr/bin/env python
# _*_ coding:utf-8 _*_

corpus = ['this is the first document.',
      'this is the second second document.',
      'and the third one.',
      'is first the third document?']
# 1)把语料库做一个分词的处理
word_list = []
for i in range(len(corpus)):
    word_list.append(corpus[i].split(' '))
print(word_list)
# 2得到每个词的id值及词频
from gensim import corpora

# 赋给语料库中每个词(不重复的词)一个整数id
dictionary = corpora.Dictionary(word_list)
new_corpus = [dictionary.doc2bow(text) for text in word_list]
print(new_corpus)

# 元组中第一个元素是词语在词典中对应的id，第二个元素是词语在文档中出现的次数
[输出]：
[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
 [(0, 1), (2, 1), (3, 1), (4, 1), (5, 2)],
 [(3, 1), (6, 1), (7, 1), (8, 1)],
 [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]]

[输入]：
# 通过下面的方法可以看到语料库中每个词对应的id
print(dictionary.token2id)
[输出]：
{'document': 0, 'first': 1, 'is': 2, 'the': 3, 'this': 4, 'second': 5, 'and': 6,
 'one': 7, 'third': 8}




















































