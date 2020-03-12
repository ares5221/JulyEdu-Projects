#!/usr/bin/env python
# _*_ coding:utf-8 _*_

corpus = ['this is the first document.',
      'this is the second second document.',
      'and the third one.',
      'is first the third document?']

word_list = []
for i in range(len(corpus)):
    word_list.append(corpus[i].split(' '))
print(word_list)

def Counter(word_list):
      count = {}
      for word in word_list:
            if word not in count:
                  count[word] = 1
            else:
                  count[word] +=1
      print('curr ',count)
      return count

countlist = []
for i in range(len(word_list)):
    count = Counter(word_list[i])
    countlist.append(count)
print(countlist)


# word可以通过count得到，count可以通过countlist得到

# count[word]可以得到每个单词的词频， sum(count.values())得到整个句子的单词总数
def tf(word, count):
      return count[word] / sum(count.values())


# 统计的是含有该单词的句子数
def n_containing(word, count_list):
      return sum(1 for count in count_list if word in count)


# len(count_list)是指句子的总数，n_containing(word, count_list)是指含有该单词的句子的总数，加1是为了防止分母为0
def idf(word, count_list):
      return math.log(len(count_list) / (1 + n_containing(word, count_list)))


# 将tf和idf相乘
def tfidf(word, count, count_list):
      return tf(word, count) * idf(word, count_list)


import math
for i, count in enumerate(countlist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, count, countlist) for word in count}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))



