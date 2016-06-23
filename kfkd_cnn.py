# coding:utf-8

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
from input_data import load2d
from collections import OrderedDict
from sklearn.cross_validation import train_test_split

# If you use gpu when using theano backend
# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'

SPECIALIST_SETTINGS = [
    dict(
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
        ),
        flip_indices=((0, 2), (1, 3)),
    ),

    dict(
        columns=(
            'nose_tip_x', 'nose_tip_y',
        ),
        flip_indices=(),
    ),

    dict(
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
        ),
        flip_indices=((0, 2), (1, 3)),
    ),

    dict(
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
        ),
        flip_indices=(),
    ),

    dict(
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
        ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
    ),

    dict(
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
        ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
    ),
]


class FlippedImageDataGenerator(ImageDataGenerator):
    flip_indices = [(0, 2), (1, 3), (4, 8), (5, 9),
                    (6, 10), (7, 11), (12, 16), (13, 17),
                    (14, 18), (15, 19), (22, 24), (23, 25)]

    def next(self):
        X_batch, y_batch = super(FlippedImageDataGenerator, self).next()
        batch_size = X_batch.shape[0]
        indices = np.random.choice(batch_size, batch_size / 2, replace=False)
        X_batch[indices] = X_batch[indices, :, :, ::-1]

        if y_batch is not None:
            y_batch[indices, ::2] = y_batch[indices, ::2] * -1

            for a, b in self.flip_indices:
                y_batch[indices, a], y_batch[indices, b] = (
                    y_batch[indices, b], y_batch[indices, a]
                )

        return X_batch, y_batch


def cnn_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, input_shape=(1, 96, 96)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(30))

    return model


def fit_model():
    start = 0.03
    stop = 0.001
    nb_epoch = 10000
    PRETRAIN = False
    learning_rate = np.linspace(start, stop, nb_epoch)

    X, y = load2d()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.2, random_state=42)

    model = cnn_model()
    if PRETRAIN:
        model.load_weights('my_cnn_model_weights.h5')
    sgd = SGD(lr=start, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)
    change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    early_stop = EarlyStopping(patience=100)

    flipgen = FlippedImageDataGenerator()
    hist = model.fit_generator(flipgen.flow(X_train, y_train),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, y_test),
                            callbacks=[change_lr, early_stop])

    model.save_weights('my_cnn_model_weights.h5', overwrite=True)
    np.savetxt('my_cnn_model_loss.csv', hist.history['loss'])
    np.savetxt('my_cnn_model_val_loss.csv', hist.history['val_loss'])


def fit_specialists():
    specialists = OrderedDict()
    start = 0.03
    stop = 0.001
    nb_epoch = 10000
    PRETRAIN = False
    learning_rate = np.linspace(start, stop, nb_epoch)

    for setting in SPECIALIST_SETTINGS:
        cols = setting['columns']
        X, y = load2d(cols=cols)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.2, random_state=42)
        model = cnn_model()
        if PRETRAIN:
            model.load_weights('my_cnn_model_weights.h5')
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.add(Dense(len(cols)))

        sgd = SGD(lr=start, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=sgd)
        lr_decay = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
        early_stop = EarlyStopping(patience=100)

        flipgen = FlippedImageDataGenerator()
        flipgen.flip_indices = setting['flip_indices']

        print('Training model for columns {} for {} epochs'.format(cols, nb_epoch))

        hist = model.fit_generator(flipgen.flow(X_train, y_train),
                                samples_per_epoch=X_train.shape[0],
                                nb_epoch=nb_epoch,
                                validation_data=(X_test, y_test),
                                callbacks=[lr_decay, early_stop])

        model.save_weights('my_cnn_model_{}_weights.h5'.format(cols[0]))
        np.savetxt('my_cnn_model_{}_loss.csv'.format(cols[0]), hist.history['loss'])
        np.savetxt('my_cnn_model_{}_val_loss.csv'.format(cols[0]), hist.history['val_loss'])

        specialists[cols] = model


def plot_loss():
    loss = np.loadtxt('my_cnn_model_loss.csv')
    val_loss = np.loadtxt('my_cnn_model_val_loss.csv')

    plt.plot(loss, linewidth=3, label='train')
    plt.plot(val_loss, linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(1e-3, 1e-2)
    plt.yscale('log')
    plt.show()


def main():
    fit_model()
    fit_specialists()

    # plot_loss()


if __name__ == '__main__':
    main()
