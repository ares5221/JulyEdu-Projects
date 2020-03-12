#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from nltk.text import TextCollection

# 首先, 把所有的文档放到TextCollection类中。
# 这个类会自动帮你断句, 做统计, 做计算
corpus = TextCollection(['this is sentence one',
                        'this is sentence two',
                        'this is sentence three'])

# 直接就能算出tfidf
# (term: 一句话中的某个term, text: 这句话)
print(corpus.tf_idf('this', 'this is sentence four'))
# 0.444342

# 同理, 怎么得到一个标准大小的vector来表示所有的句子?

# 对于每个新句子
new_sentence = 'this is sentence five'
# 遍历一遍所有的vocabulary中的词:
for word in standard_vocab:
    print(corpus.tf_idf(word, new_sentence))
    # 我们会得到一个巨长(=所有vocab长度)的向量
