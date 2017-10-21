import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import re
import bs4
import gensim
import nltk
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing import sequence, text
import pickle


def load_data(train_file, test_file, extra_file):
    train_set = pd.read_csv(train_file, header=0, sep='\t')
    test_set = pd.read_csv(test_file, header=0, sep='\t')
    extra_set = pd.read_csv(extra_file, header=0, sep='\t', error_bad_lines=False, warn_bad_lines=True)
    print('Train set info:')
    train_set.info()
    print('Test set info:')
    test_set.info()
    print('Extra set info:')
    extra_set.info()

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
    df[column] = df[column].map(lambda x: ' '.join(x))
    return df


def texts_to_sequences(df, column, tokenizer, maxlen=300):
    seq = tokenizer.texts_to_sequences(line for line in df[column].values)
    seq = sequence.pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    return seq


train_set, test_set, extra_set = load_data(r'E:\OpenSourceDatasetCode\Dataset\Bag of Words Meets Bags of Popcorn\labeledTrainData.tsv',
                                           r'E:\OpenSourceDatasetCode\Dataset\Bag of Words Meets Bags of Popcorn\testData.tsv',
                                           r'E:\OpenSourceDatasetCode\Dataset\Bag of Words Meets Bags of Popcorn\unlabeledTrainData.tsv')

train_df = raw_to_texts(train_set, 'review', remove_stopwords=True)
test_df = raw_to_texts(test_set, 'review', remove_stopwords=True)
extra_df = raw_to_texts(extra_set, 'review', remove_stopwords=True)
all_reviews = pd.concat([train_df, test_df, extra_df], ignore_index=True)

sentences = [line for line in all_reviews['review'].values]
model = gensim.models.Word2Vec.load(r'data\wiki_en_model')
model.train(sentences, total_examples=len(sentences), epochs=3)
model.save(r'data\wiki_en_IMDB_model')
model.wv.save_word2vec_format(r'data\wiki_en_IMDB_model.bin', binary=True)
del model

texts = [line for line in all_reviews['review'].values]
sequence_tokenizer = text.Tokenizer()
sequence_tokenizer.fit_on_texts(texts)
dic_len = len(sequence_tokenizer.word_index)

train_X = texts_to_sequences(train_df, 'review', sequence_tokenizer, maxlen=1500)
test_X = texts_to_sequences(test_df, 'review', sequence_tokenizer, maxlen=1500)
labels = np.array(train_df.sentiment, dtype=int)
train_Y = to_categorical(labels, num_classes=2)
del train_df, test_df, extra_df

model = gensim.models.KeyedVectors.load_word2vec_format(r'data\wiki_en_IMDB_model.bin', binary=True)
model.init_sims(replace=True)
word_vec_dims = 300
W = np.random.uniform(-0.25, 0.25, (dic_len + 1, word_vec_dims))
W[0] = np.zeros((word_vec_dims,), dtype=int)
for word, index in sequence_tokenizer.word_index.items():
    if word in model.vocab:
        W[index, :] = model[word]

del model
del sequence_tokenizer

with open('IMDB_train_test_data.pkl', 'wb') as file:
    pickle.dump([train_X, train_Y, test_X, W], file)

file.close()
