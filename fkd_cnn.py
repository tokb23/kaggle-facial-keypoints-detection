# coding:utf-8

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
# from keras.models import model_from_json

from input_data import load2d
import numpy as np
from collections import OrderedDict
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import json

# if you use gpu
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


def load_model():
    # model = model_from_json(open('my_cnn_model_architecture.json').read())
    # model.load_weights('my_cnn_model_weights.h5')
    f_loss = open('my_cnn_model_loss_history.json')
    f_val_loss = open('my_cnn_model_val_loss_history.json')
    loss = json.load(f_loss)
    val_loss = json.load(f_val_loss)
    print('successfully loaded')

    return loss, val_loss


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
    nb_epoch = 3000
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
    lr_decay = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    early_stop = EarlyStopping(patience=100)

    flipgen = FlippedImageDataGenerator()
    hist = model.fit_generator(flipgen.flow(X_train, y_train),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, y_test),
                            callbacks=[lr_decay, early_stop])

    plot_loss(hist)
    save_model(model, hist)


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

        specialists[cols] = model

        # save history


def plot_loss(loss=None, val_loss=None, hist=None):
    if hist is not None:
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
    plt.plot(loss, linewidth=3, label='train')
    plt.plot(val_loss, linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(1e-4, 1e-2)
    plt.yscale('log')
    plt.show()


def save_model(model=None, hist=None):
    if model is not None:
        # json_model = model.to_json()
        # open('my_cnn_model_architecture.json', 'w').write(json_model)
        model.save_weights('my_cnn_model_weights.h5', overwrite=True)

    if hist is not None:
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        f_loss = open('my_cnn_model_loss_history.json', 'w')
        f_val_loss = open('my_cnn_model_val_loss_history.json', 'w')
        json.dump(loss, f_loss)
        json.dump(val_loss, f_val_loss)

    print('successfully saved')


def main():
    fit_model()
    fit_specialists()

    # loss, val_loss = load_model()
    # plot_loss(loss, val_loss)


if __name__ == '__main__':
    main()
