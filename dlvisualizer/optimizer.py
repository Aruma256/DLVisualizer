import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from keras.models import Model
from keras.layers import Input, Dense
from keras.initializers import Constant
from keras.optimizers import *

from test_functions import *

TESTFUNCTION = Beale
FRAMES = 2000
FRAME_SKIP = 10
OPTIMIZERS = [
    ('SGD', SGD()),
    ('MomentumSGD', SGD(momentum=0.99)),
    ('Adagrad', Adagrad()),
    ('Adadelta', Adadelta()),
    ('RMSprop', RMSprop()),
    ('Adam', Adam()),
    ('Adamax', Adamax()),
    ('Nadam', Nadam()),
]


def get_models():
    models = []
    for name, optimizer in OPTIMIZERS:
        input = Input((1,))
        dense = Dense(2,
                      kernel_initializer=Constant(TESTFUNCTION.init_pos),
                      use_bias=False)
        model = Model(input, dense(input), name=name)
        model.compile(optimizer=optimizer,
                      loss=lambda y_true, y_pred: TESTFUNCTION(y_pred[0][0], y_pred[0][1]))
        models.append(model)

    return models


def main():
    FAKE_INPUT = np.ones((1, 1))
    FAKE_OUTPUT = np.zeros((1, 2))
    models = get_models()
    testfunction = TESTFUNCTION
    trajectories = [([], []) for _ in range(len(models))]

    def plot(step, *args, **kwargs):
        plt.cla()
        plt.contour(*testfunction.contour,
                    40,
                    levels=testfunction.contour_levels)
        for i, model in enumerate(models):
            x, y = model.layers[1].get_weights()[0][0]
            trajectories[i][0].append(x)
            trajectories[i][1].append(y)
            for _ in range(FRAME_SKIP):
                model.train_on_batch(FAKE_INPUT, FAKE_OUTPUT)
            plt.plot(trajectories[i][0],
                     trajectories[i][1],
                     '-o',
                     label=model.name, markevery=[-1])
        plt.plot(*testfunction.minimum_pos, 'o', label='Minimum')
        plt.grid()
        plt.legend()
        plt.xlim(testfunction.x_range)
        plt.ylim(testfunction.y_range)

    anim = anm.FuncAnimation(plt.figure(), plot,
                             interval=1, frames=FRAMES, init_func=lambda: None)
    #anim.save("Sample.mp4", writer=anm.FFMpegFileWriter(fps=120))
    plt.show()


if __name__ == '__main__':
    main()
