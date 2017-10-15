from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam, Adagrad
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
import numpy as np
import pandas as pd


def load_data():
    # Load data
    train = pd.read_csv(r'E:\OpenSourceDatasetCode\Dataset\MNIST\train.csv')
    test = pd.read_csv(r'E:\OpenSourceDatasetCode\Dataset\MNIST\test.csv')
    Y_train = train['label']
    # Drop 'label' column
    X_train = train.drop(labels=['label'], axis=1)
    # free some space
    del train

    # Normalization
    X_train /= 255.0
    test /= 255.0
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)
    Y_train = to_categorical(Y_train, num_classes=10)
    # Set the random seed
    # random_seed = 2
    # Split the train and the validation set for the fitting
    # X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

    return X_train, Y_train, test


def get_model(drop1=0.25, drop2=0.5, lr=0.001):
    if K.image_dim_ordering() == 'tf':
        inp = Input(shape=(28, 28, 1))
        bn_axis = 3
    else:
        inp = Input(shape=(1, 28, 28))
        bn_axis = 1

    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='Same', input_shape=(28, 28, 1)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='Same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='Same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D((2, 2), (2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    K.set_image_dim_ordering('tf')
    X_train, Y_train, test = load_data()
    model = get_model()
    lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    epochs = 30
    batch_size = 128
    # With data augmentation to prevent overfitting
    data_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
    )
    data_gen.fit(X_train)

    history = model.fit_generator(data_gen.flow(X_train, Y_train, batch_size=batch_size),
                                  epochs=epochs, validation_data=(X_val, Y_val),
                                  verbose=2, steps_per_epoch=X_train.shape[0] // batch_size,
                                  callbacks=[lr_reduction])

    # Predict results
    results = model.predict(test)
    results = np.argmax(results, axis=1)
    results = pd.Series(results, name='Label')
    submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)
    submission.to_csv('LeNet_MNIST_datagen.csv', index=False)