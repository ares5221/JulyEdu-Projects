#!/usr/bin/env python
# _*_ coding:utf-8 _*_

corpus = ['this is the first document.',
      'this is the second second document.',
      'and the third one.',
      'is first the third document?']
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(corpus)

# 得到语料库所有不重复的词
print(tfidf_vec.get_feature_names())

# 得到每个单词对应的id值
print(tfidf_vec.vocabulary_)

# 得到每个句子所对应的向量
# 向量里数字的顺序是按照词语的id顺序来的
print(tfidf_matrix.toarray())
















































