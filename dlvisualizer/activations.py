import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import *


def main():
    X_RANGE = (-4, 4)
    x = np.linspace(*X_RANGE, 100)
    ACTIVATIONS = [
        ('sigmoid', Activation('sigmoid')),
        ('tanh', Activation('tanh')),
        ('ReLU', Activation('relu')),
        ('LeakyReLU', LeakyReLU()),
        ('PReLU', PReLU()),
        ('ELU', Activation('elu')),
        ('SELU', Activation('selu'))
    ]

    input = Input(x.shape)
    for name, layer in ACTIVATIONS:
        model = Model(input, layer(input))
        y = model.predict_on_batch(np.array([x]))[0]
        plt.plot(x, y, label=name)

    plt.grid()
    plt.legend()
    plt.xlim(X_RANGE)

    plt.show()
    #plt.savefig('activations.png')


if __name__ == '__main__':
    main()
