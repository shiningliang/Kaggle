import time
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Input, merge
from keras.models import Model


def get_model(W, seq_len, conv_filters, kernel_size=3, hidden_size=100):
    input = Input(shape=(seq_len,), dtype='int32')
    embedded = Embedding(input_dim=W.shape[0], output_dim=W.shape[1], input_length=seq_len, weights=[W])(input)
    embedded = Dropout(0.5)(embedded)

    tensor = Conv1D(filters=conv_filters, kernel_size=3)(embedded)
    tensor = Activation('relu')(tensor)

    forwards = GRU(units=hidden_size)(tensor)
    backwards = GRU(units=hidden_size, go_backwards=True)(tensor)

    output = merge([forwards, backwards], mode='concat', concat_axis=1)
    # output = Flatten()(output)
    output = Dropout(0.5)(output)
    output = Dense(2, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(output)
    output = Activation('softmax')(output)

    model = Model(input, output)
    print(model.summary())
    return model

class Configs():
    conv_filters = 256
    kernel_size = 3
    rnn_hidden_units = 256
    drop_prob = 0.5
    l2_reg = 0.001
    num_epochs = 20
    batch_size = 128


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # Load data
    print('Loading data...')
    with open('IMDB_train_test_data.pkl', 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    train_X, train_Y, test_X, W = data[0], data[1], data[2], data[3]
    print('Data loaded!')

    # Build model
    config = Configs()
    model = get_model(W, train_X.shape[1], config.conv_filters, config.kernel_size, config.rnn_hidden_units)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    print('Start train at ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    model.fit(train_X, train_Y, batch_size=config.batch_size, epochs=config.num_epochs, verbose=2, shuffle=True)
    print('End train at ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    model.save_weights('c_lstm_' + str(config.num_epochs) + 'epochs.h5')

    # model.load_weights('c_lstm_20epochs.h5')
    pred = model.predict(test_X)
    pred = (pred > 0.5).astype('int32')
    test_set = pd.read_csv('testData.tsv', header=0, sep='\t')
    submission = pd.DataFrame({'id': test_set['id'], 'sentiment': pred[:, 0]})
    submission.to_csv('C_BiLSTM_pred.csv', index=False)
