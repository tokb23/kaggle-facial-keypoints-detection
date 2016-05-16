from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.models import model_from_json

from input_data import load2d
import numpy as np
from collections import OrderedDict
from copy import deepcopy

LOAD = False
data_argumentation = True
train_specialist = False
fname_pretrain = False

flip_indices = [(0, 2), (1, 3), (4, 8), (5, 9), 
                (6, 10), (7, 11), (12, 16), (13, 17), 
                (14, 18), (15, 19), (22, 24), (23, 25)]

lr_start = 0.03 # initial value of learning rate 
lr_stop = 0.0001 # final value of learning rate
m_start = 0.9 # initial value of momentum
m_stop = 0.99 # final value of momentum
nb_epoch = 10 # number of epochs for training
batch_size=32

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


def fit_specialists(model):
    specialists = OrderedDict()

    for setting in SPECIALIST_SETTINGS:
        cols = setting['columns']
        X, y = load2d(cols=cols)
        nb_epoch = int(1e7 / y.shape[0])

        model_special = deepcopy(model)
        model_special.add(Dense(y.shape[1], init='glorot_uniform',))
        model_special.add(Activation('linear'))

        if fname_pretrain:
            model_special.load_weights('./model/my_cnn_model_weights.h5')
            print('successfully loaded')

        sgd = SGD(lr=0.03, decay=0.0, momentum=0.9, nesterov=True)
        model_special.compile(loss='mse', optimizer=sgd)

        early_stop = EarlyStopping(monitor='val_loss', patience=200, verbose=1, mode='auto')

        lr_decay = np.linspace(lr_start, lr_stop, nb_epoch)
        m_grow = np.linspace(m_start, m_stop, nb_epoch)

        print("Training model for columns {} for {} epochs".format(cols, nb_epoch))

        datagen = ImageDataGenerator(horizontal_flip=True, flip_indices=setting['flip_indices'])
        for i in range(nb_epoch):
            sgd.lr = lr_decay[i]
            sgd.momentum = m_grow[i]
            model_special.fit_generator(datagen.flow(X, y), samples_per_epoch=X.shape[0], nb_epoch=1, verbose=1, callbacks=[early_stop])

        specialists[cols] = model_special

    return specialists


def load_model():
    model = model_from_json(open('./model/my_cnn_model_architecture.json').read())
    model.load_weights('./model/my_cnn_model_weights.h5')
    print('successfully loaded')

    return model


def cnn_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, init='glorot_uniform', border_mode='valid', input_shape=(1, 96, 96)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64, 2, 2, init='glorot_uniform', border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 2, 2, init='glorot_uniform', border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1000, init='glorot_uniform',))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, init='glorot_uniform',))
    model.add(Activation('relu'))
    model.add(Dense(30, init='glorot_uniform',))
    model.add(Activation('linear'))

    return model


def train(model):
    sgd = SGD(lr=0.03, decay=0.0, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)

    early_stop = EarlyStopping(monitor='val_loss', patience=200, verbose=1, mode='auto')

    #lr_decay = np.linspace(lr_start, lr_stop, nb_epoch)
    #m_grow = np.linspace(m_start, m_stop, nb_epoch)

    X, y = load2d()
    nb_sample = X.shape[0]

    if data_argumentation:
        for i in range(nb_epoch):
            perm = np.arange(nb_sample)
            np.random.shuffle(perm)
            X = X[perm]
            y = y[perm]
            #sgd.lr = lr_decay[i]
            #sgd.momentum = m_grow[i]
            print("[epoch %d]" % int(i+1))
            for j in range(nb_sample / batch_size):
                print "[batch %d]" % int(j+1)
                start = j * batch_size
                end = (j+1) * batch_size
                X_arg, y_arg = flip_data(X[start:end], y[start:end], flip_indices)
                hist = model.fit(X_arg, y_arg, nb_epoch=1, verbose=1, validation_split=0.2, callbacks=[early_stop])
    else:
        for i in range(nb_epoch):
            #sgd.lr = lr_decay[i]
            #sgd.momentum = m_grow[i]
            hist = model.fit(X, y, nb_epoch=1, verbose=1, validation_split=0.2, callbacks=[early_stop])

    return model, hist


def flip_data(X, y, flip_indices):
    bs = X.shape[0]
    indices = np.random.choice(bs, bs / 2, replace=False)
    X[indices] = X[indices, :, :, ::-1]
    y[indices, ::2] = y[indices, ::2] * -1
    for a, b in flip_indices:
        y[indices, a], y[indices, b] = (y[indices, b], y[indices, a])

    return X, y


def save_model(model):
    json_string = model.to_json()
    open('./model/my_cnn_model_architecture.json', 'w').write(json_string)
    model.save_weights('./model/my_cnn_model_weights.h5')
    print('successfully saved')


def main():
    if LOAD:
        model = load_model()
    else:
        model = cnn_model()

    if train_specialist:
        train_model = fit_specialists(model)
    else:
        train_model, _ = train(model)

    save_model(train_model)


if __name__ == '__main__':
    main()
