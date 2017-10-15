from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import random
import re

import bs4
import keras.preprocessing.sequence
import nltk
import numpy

seed = 42
random.seed(seed)
numpy.random.seed(seed)

wordnet = nltk.stem.WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words('english'))


def raw_to_words(df, column, remove_stopwords=False, dictionary=None):
    df[column] = df[column].map(lambda x: bs4.BeautifulSoup(x, 'lxml').get_text())
    df[column] = df[column].map(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))
    df[column] = df[column].map(lambda x: x.lower().split())
    df[column] = df[column].map(lambda x: [wordnet.lemmatize(y) for y in x])
    if remove_stopwords:
        df[column] = df[column].map(lambda x: [y for y in x if y not in stopwords])
    if dictionary:
        df[column] = df[column].map(lambda x: [y for y in x if y in dictionary])
    return df


def raw_to_texts(df, column, remove_stopwords=False, dictionary=None):
    df = raw_to_words(df, column, remove_stopwords, dictionary)
    df[column] = df[column].map(lambda x: ' '.join(x))
    return df


def texts_to_sequences(df, column, tokenizer, maxlen=300):
    seq = tokenizer.texts_to_sequences(line.encode('utf-8') for line in df[column].values)
    print('mean:', numpy.mean([len(x) for x in seq]))
    print('std:', numpy.std([len(x) for x in seq]))
    print('median:', numpy.median([len(x) for x in seq]))
    print('max:', numpy.max([len(x) for x in seq]))
    seq = keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    return seq