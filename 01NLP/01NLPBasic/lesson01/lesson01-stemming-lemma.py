#!/usr/bin/env python
# _*_ coding:utf-8 _*_

# Stemming
# 词干提取：一般来说，就是把不影响词性的inflection的小尾巴砍掉
#
# walking
# 砍ing = walk
# walked
# 砍ed = walk

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
print(porter_stemmer.stem('maximum'))
print(porter_stemmer.stem('presumably'))
print( porter_stemmer.stem('multiply'))
print(porter_stemmer.stem('provision'))
print('\nLancasterStemmer:')
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
print(lancaster_stemmer.stem('maximum'))
print( lancaster_stemmer.stem('presumably'))
print( lancaster_stemmer.stem('presumably'))
print('\nSnowballStemmer:')
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')
print( snowball_stemmer.stem('maximum'))
print( snowball_stemmer.stem('presumably'))
print('\nPorterStemmer:')
from nltk.stem.porter import PorterStemmer
p = PorterStemmer()
print(p.stem('went'))
print( p.stem('wenting'))

# Lemmatization 词形归一：把各种类型的词的变形，都归为一个形式
# 
# went 归一 = go
# are 归一 = be
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
print(wordnet_lemmatizer.lemmatize('dogs'))
print(wordnet_lemmatizer.lemmatize('churches'))
print(wordnet_lemmatizer.lemmatize('aardwolves'))
print(wordnet_lemmatizer.lemmatize('abaci'))
print(wordnet_lemmatizer.lemmatize('hardrock'))

# lemma pro
# 木有POS Tag，默认是NN 名词
print(wordnet_lemmatizer.lemmatize('are'))
print(wordnet_lemmatizer.lemmatize('is'))

# 加上POS Tag
print(wordnet_lemmatizer.lemmatize('is', pos='v'))
print(wordnet_lemmatizer.lemmatize('are', pos='v'))

# NLTK标注POS Tag
text = nltk.word_tokenize('what does the fox say')
print(nltk.pos_tag(text))

#  test stopwords
from nltk.corpus import stopwords
word_list = 'life is like a box of chocolate'
filtered_words = [word for word in word_list if word not in stopwords.words('english')]
print(filtered_words)
