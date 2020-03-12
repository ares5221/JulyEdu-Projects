#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# 情感分析
import nltk
sentiment_dictionary = {}
for line in open('data/AFINN-111.txt'):
    word, score = line.split('\t')
    sentiment_dictionary[word] = int(score)
print(sentiment_dictionary)
# 把这个打分表记录在一个Dict上以后
# 跑一遍整个句子，把对应的值相加
words = 'this movie is very good'
words = nltk.word_tokenize(words)
print(words)
total_score =sum(sentiment_dictionary.get(word, 0) for word in words)
print(total_score)

# total_score = 0
# for word in words:
#     print(word)
#     print(sentiment_dictionary.get(word))
#     if word in sentiment_dictionary:
#         total_score += sentiment_dictionary.get(word)
#     else:
#         total_score += 0
# print(total_score)
