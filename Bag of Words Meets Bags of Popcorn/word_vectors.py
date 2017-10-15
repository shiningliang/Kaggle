from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import gensim
import keras.preprocessing.text
import nltk
import numpy
import pandas

import data
import init

word_vec_dim = 300


def build_word2vec():
    sentences = []
    for row in data.train_df['review'].append(data.unlabeled_df['review']):
        sentences_df = pandas.DataFrame(nltk.sent_tokenize(row.decode('utf-8').strip()), columns=['sentence'])
        sentences_df = init.raw_to_words(sentences_df, 'sentence')
        sentences += sentences_df['sentence'].tolist()

    model = gensim.models.Word2Vec(sentences, size=word_vec_dim, window=10, min_count=1, workers=1, seed=init.seed)
    return model


# word2vec = build_word2vec()
word2vec = gensim.models.Word2Vec.load_word2vec_format('./init/300features_10contexts.bin', binary=True)
word2vec.init_sims(replace=True)

del data.unlabeled_df
train_df = init.raw_to_texts(data.train_df, 'review', dictionary=word2vec.vocab)
del data.train_df
test_df = init.raw_to_texts(data.test_df, 'review', dictionary=word2vec.vocab)
del data.test_df

sequence_tokenizer = keras.preprocessing.text.Tokenizer()
sequence_tokenizer.fit_on_texts(line.encode('utf-8') for line in train_df['review'].values) # texts：要用以训练的文本列表

max_features = len(sequence_tokenizer.word_index)

train = init.texts_to_sequences(train_df, 'review', sequence_tokenizer, maxlen=2500) # 文本转换为字典序列
del train_df
test = init.texts_to_sequences(test_df, 'review', sequence_tokenizer, maxlen=2500)
del test_df

weights = numpy.zeros((max_features + 1, word_vec_dim))
for word, index in sequence_tokenizer.word_index.items():
    # if index <= max_features and word in word2vec.vocab:
    #     weights[index, :] = word2vec[word]
    if word in word2vec.vocab:
        weights[index, :] = word2vec[word]
    else:
        weights[index, :] = numpy.random.uniform(-0.25, 0.25, word_vec_dim)
del word2vec
del sequence_tokenizer