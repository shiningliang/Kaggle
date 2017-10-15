import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from keras.layers.merge import add
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
import numpy as np


sns.set(style='white', context='notebook', palette='deep')

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
    random_seed = 2
    # Split the train and the validation set for the fitting
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

    return X_train, X_val, Y_train, Y_val, test


# 输入输出大小相同
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    the identity_block is the block that has no conv layer at shortcut
    :param x:
    :param nb_filter:
    :param kernel_size:
    :return:
    """
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    k1, k2, k3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    out = Conv2D(k1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(out)
    out = Activation('relu')(out)

    # border_name=same 卷积层输入输出shape相同
    out = Conv2D(k2, kernel_size, padding='same', name=conv_name_base + '2b')(out)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)

    out = Conv2D(k3, (1, 1), name=conv_name_base + '2c')(out)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(out)

    out = add([out, input_tensor])
    out = Activation('relu')(out)

    return out


# 输出比输入小
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    conv_block is the block that has a conv layer at shortcut
    params:
    input_tensor: input tensor
    kernel_size: defualt 3, the kernel size of middle conv layer at main path
    filters: list of integers, the nb_filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should has subsample=(2,2) as well
    """
    k1, k2, k3 = filters
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    out = Conv2D(k1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(out)
    out = Activation('relu')(out)

    out = Conv2D(k2, kernel_size, padding='same', name=conv_name_base + '2b')(out)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)

    out = Conv2D(k3, (1, 1), name=conv_name_base + '2c')(out)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(out)

    shortcut = Conv2D(k3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    out = add([out, shortcut])
    out = Activation('relu')(out)

    return out


def get_model(include_top=True, pooling=None):
    if K.image_dim_ordering() == 'tf':
        inp = Input(shape=(28, 28, 1))
        bn_axis = 3
    else:
        inp = Input(shape=(1, 28, 28))
        bn_axis = 1

    out = conv_block(inp, 3, [128, 128, 512], stage=1, block='a')
    # conv_name_base = 'res' + str(1) + 'a' + '_branch'
    # bn_name_base = 'bn' + str(1) + 'a' + '_branch'
    #
    # out = Conv2D(128, (1, 1), strides=(2,2), name=conv_name_base + '2a')(inp)
    # out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(out)
    # out = Activation('relu')(out)
    #
    # out = Conv2D(128, 3, padding='same', name=conv_name_base + '2b')(out)
    # out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(out)
    # out = Activation('relu')(out)
    #
    # out = Conv2D(512, (1, 1), name=conv_name_base + '2c')(out)
    # out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(out)
    #
    # shortcut = Conv2D(512, (1, 1), strides=(2,2), name=conv_name_base + '1')(inp)
    # shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    #
    # out = add([out, shortcut])
    # out = Activation('relu')(out)

    out = identity_block(out, 3, [128, 128, 512], stage=1, block='b')
    out = identity_block(out, 3, [128, 128, 512], stage=1, block='c')
    out = identity_block(out, 3, [128, 128, 512], stage=1, block='d')

    out = conv_block(out, 3, [256, 256, 1024], stage=2, block='a')
    out = identity_block(out, 3, [256, 256, 1024], stage=2, block='b')
    out = identity_block(out, 3, [256, 256, 1024], stage=2, block='c')
    out = identity_block(out, 3, [256, 256, 1024], stage=2, block='d')
    out = identity_block(out, 3, [256, 256, 1024], stage=2, block='e')
    out = identity_block(out, 3, [256, 256, 1024], stage=2, block='f')

    out = conv_block(out, 3, [512, 512, 2048], stage=3, block='a')
    out = identity_block(out, 3, [512, 512, 2048], stage=3, block='b')
    out = identity_block(out, 3, [512, 512, 2048], stage=3, block='c')

    out = AveragePooling2D((4, 4), name='avg_pool')(out)

    if include_top:
        out = Flatten()(out)
        out = Dense(10, activation='softmax', name='fc10')(out)
    else:
        if pooling == 'avg':
            out = GlobalAveragePooling2D()(out)
        else:
            out = GlobalMaxPooling2D()(out)

    model = Model(inp, out)
    print(model.summary())
    return model


if __name__ == '__main__':
    K.set_image_dim_ordering('tf')
    X_train, X_val, Y_train, Y_val, test = load_data()
    model = get_model(True, 'avg')
    optimizer = RMSprop()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
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