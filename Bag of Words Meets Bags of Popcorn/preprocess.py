import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import random
import re
import bs4
import gensim
import nltk
import numpy as np
import pandas as pd
from keras.preprocessing import sequence, text


def load_data(train_file, test_file, extra_file):
    train_set = pd.read_csv(train_file, header=0, sep='\t')
    test_set = pd.read_csv(test_file, header=0, sep='\t')
    extra_set = pd.read_csv(extra_file, header=0, sep='\t', error_bad_lines=False, warn_bad_lines=True)
    train_set.info()
    test_set.info()

    return train_set, test_set, extra_set


def raw_to_words(df, column, remove_stopwords=False):
    wordnet = nltk.stem.WordNetLemmatizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))

    df[column] = df[column].map(lambda x: bs4.BeautifulSoup(x, 'html.parser').get_text())
    df[column] = df[column].map(lambda x: re.sub(r'[^a-zA-Z,!?\'\`]', ' ', x))
    df[column] = df[column].map(lambda x: x.lower().split())
    df[column] = df[column].map(lambda x: [wordnet.lemmatize(y) for y in x])
    if remove_stopwords:
        df[column] = df[column].map(lambda x: [y for y in x if y not in stopwords])

    df['num_words'] = df[column].apply(lambda x: len(x))

    return df


def raw_to_texts(df, column, remove_stopwords=False):
    df = raw_to_words(df, column, remove_stopwords)
    print(np.max(df['num_words']))
    df[column] = df[column].map(lambda x: ' '.join(x))
    return df


def texts_to_sequences(df, column, tokenizer, maxlen=300):
    seq = tokenizer.texts_to_sequences(line.encode('utf-8') for line in df[column].values)
    print('mean:', np.mean([len(x) for x in seq]))
    print('std:', np.std([len(x) for x in seq]))
    print('median:', np.median([len(x) for x in seq]))
    print('max:', np.max([len(x) for x in seq]))
    seq = sequence.pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    return seq


train_set, test_set, extra_set = load_data(r'E:\OpenSourceDatasetCode\Dataset\Bag of Words Meets Bags of Popcorn\labeledTrainData.tsv',
                                           r'E:\OpenSourceDatasetCode\Dataset\Bag of Words Meets Bags of Popcorn\testData.tsv',
                                           r'E:\OpenSourceDatasetCode\Dataset\Bag of Words Meets Bags of Popcorn\unlabeledTrainData.tsv')

train_df = raw_to_texts(train_set, 'review', remove_stopwords=True)
test_df = raw_to_texts(test_set, 'review', remove_stopwords=True)
extra_df = raw_to_texts(extra_set, 'review', remove_stopwords=True)

sequence_tokenizer = text.Tokenizer()
sequence_tokenizer.fit_on_texts(line.encode('utf-8') for line in train_df['review'].values) # texts：要用以训练的文本列表
