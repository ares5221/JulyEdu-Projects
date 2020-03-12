#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf
from tensorflow import keras

from sklearn.cluster import KMeans


class ASCIIAutoencoder():
    """基于字符的Autoencoder."""
    def __init__(self, sen_len=512, encoding_dim=32, epoch=50, val_ratio=0.3):
        """
        Init.
        :param sen_len: 把sentences pad成相同的长度
        :param encoding_dim: 压缩后的维度dim
        :param epoch: 要跑多少epoch
        :param kmeanmodel: 简单的KNN clustering模型
        """
        self.sen_len = sen_len
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.encoder = None
        self.kmeanmodel = KMeans(n_clusters=2)
        self.epoch = epoch

    def fit(self, x):
        """
        模型构建。
        :param x: input text
        """
        # 把所有的trainset都搞成同一个size，并把每一个字符都换成ascii码
        x_train = self.preprocess(x, length=self.sen_len)
        # 然后给input预留好位置
        input_text = keras.layers.Input(shape=(self.sen_len,))
        # "encoded" 每一经过一层，都被刷新成小一点的“压缩后表达式”
        encoded = keras.layers.Dense(1024, activation='tanh')(input_text)
        encoded = keras.layers.Dense(512, activation='tanh')(encoded)
        encoded = keras.layers.Dense(128, activation='tanh')(encoded)
        encoded = keras.layers.Dense(self.encoding_dim, activation='tanh')(encoded)

        # "decoded" 就是把刚刚压缩完的东西，给反过来还原成input_text
        decoded = keras.layers.Dense(128, activation='tanh')(encoded)
        decoded = keras.layers.Dense(512, activation='tanh')(decoded)
        decoded = keras.layers.Dense(1024, activation='tanh')(decoded)
        decoded = keras.layers.Dense(self.sen_len, activation='sigmoid')(decoded)

        # 整个从大到小再到大的model，叫 autoencoder
        self.autoencoder = keras.models.Model(input=input_text, output=decoded)

        # 那么 只从大到小（也就是一半的model）就叫 encoder
        self.encoder = keras.models.Model(input=input_text, output=encoded)


   # 同理，我们接下来搞一个decoder出来，也就是从小到大的model
        # 来，首先encoded的input size给预留好
        encoded_input =  keras.layers.Input(shape=(1024,))
        # autoencoder的最后一层，就应该是decoder的第一层
        decoder_layer = self.autoencoder.layers[-1]
        # 然后我们从头到尾连起来，就是一个decoder了！
        decoder = keras.models.Model(input=encoded_input, output=decoder_layer(encoded_input))

        # compile
        self.autoencoder.compile(optimizer='adam', loss='mse')

        # 跑起来
        self.autoencoder.fit(x_train, x_train,
                             nb_epoch=self.epoch,
                             batch_size=1000,
                             shuffle=True,
                             )

        # 这一部分是自己拿自己train一下KNN，一件简单的基于距离的分类器
        x_train = self.encoder.predict(x_train)
        self.kmeanmodel.fit(x_train)
