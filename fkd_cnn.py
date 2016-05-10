from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from input_data import load2d

from keras.models import model_from_json

LOAD = False

def load_model():
    model = model_from_json(open('./model/my_cnn_model_architecture.json').read())
    model.load_weights('./model/my_cnn_model_weights.h5')
    sgd = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)
    print('successfully loaded')

    return model

def cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, init='glorot_uniform', border_mode='valid', input_shape=(1, 96, 96)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 2, 2, init='glorot_uniform', border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 2, 2, init='glorot_uniform', border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, init='glorot_uniform',))
    model.add(Activation('relu'))
    model.add(Dense(500, init='glorot_uniform',))
    model.add(Activation('relu'))
    model.add(Dense(30, init='glorot_uniform',))
    model.add(Activation('linear'))

    sgd = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)

    X, y = load2d()
    hist = model.fit(X, y, nb_epoch=10, verbose=1, validation_split=0.2)

    return model, hist

def save_model(model):
    json_string = model.to_json()
    open('./model/my_cnn_model_architecture.json', 'w').write(json_string)
    model.save_weights('./model/my_cnn_model_weights.h5')
    print('successfully saved')

def main():
    if LOAD:
        model = load_model()
    else:
        model, hist = cnn_model()
        save_model(model)

if __name__ == '__main__':
    main()
