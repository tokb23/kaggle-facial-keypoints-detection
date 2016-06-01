# coding:utf-8

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

from input_data import load
import matplotlib.pyplot as plt


def nn_model():
    model = Sequential()
    model.add(Dense(100, input_dim=9216))
    model.add(Activation('relu'))
    model.add(Dense(30))

    return model


def plot_loss(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss, linewidth=3, label='train')
    plt.plot(val_loss, linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(1e-3, 1e-2)
    plt.yscale('log')
    plt.show()


def check_test(model):
    def plot_sample(x, y, axis):
        img = x.reshape(96, 96)
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

    X, _ = load(test=True)
    y_pred = model.predict(X)

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred[i], ax)

    plt.show()


def save_model(model):
    # json_string = model.to_json()
    # open('my_nn_model_architecture.json', 'w').write(json_string)
    model.save_weights('my_nn_model_weights.h5')
    print('successfully saved')


def main():
    PRETRAIN = False

    X, y = load()

    model = nn_model()
    if PRETRAIN:
        model.load_weights('my_nn_model_weights.h5')
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)

    hist = model.fit(X, y, nb_epoch=1000, verbose=1, validation_split=0.2)

    plot_loss(hist)
    check_test(model)
    save_model(model)


if __name__ == '__main__':
    main()
