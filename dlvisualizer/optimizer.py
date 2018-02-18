import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from keras.models import Model
from keras.layers import Input, Dense
from keras.initializers import Constant
from keras.optimizers import *

class Loss:
  def __init__(self, f, init_pos, minimum_pos, x_range, y_range, contour_levels=None):
    self.f = f
    self.init_pos = init_pos
    self.minimum_pos = minimum_pos
    self.x_range = x_range
    self.y_range = y_range
    self.contour_levels = contour_levels

beale = Loss(lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2,
             init_pos=(1, 1),
             minimum_pos=(3, 0.5),
             x_range=(-4.5, 4.5),
             y_range=(-4.5, 4.5),
             contour_levels=np.linspace(0, 50, 20))

LOSS = beale

CONTOUR = [np.linspace(*LOSS.x_range, 100), np.linspace(*LOSS.y_range, 100)]
CONTOUR = list(np.meshgrid(*CONTOUR))
CONTOUR.append(LOSS.f(*CONTOUR))

FAKE_INPUT = np.ones((1, 1))
FAKE_OUTPUT = np.zeros((1, 2))

FRAMES = 2000
SKIP = 10
OPTIMIZERS = [
  ('SGD', SGD()),
  ('MomentumSGD', SGD(momentum=0.9)),
  ('Adagrad', Adagrad()),
  ('Adadelta', Adadelta()),
  ('RMSprop', RMSprop()),
  ('Adam', Adam()),
  ('Adamax', Adamax()),
  ('Nadam', Nadam()),
]

TRAJECTORY = []
MODELS = []
for name, optimizer in OPTIMIZERS:
  input = Input((1,))
  dense = Dense(2, kernel_initializer=Constant(LOSS.init_pos), use_bias=False)
  model = Model(input, dense(input), name=name)
  model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: LOSS.f(y_pred[0][0], y_pred[0][1]))
  MODELS.append(model)

TRAJECTORY = [(np.zeros(FRAMES), np.zeros(FRAMES)) for _ in range(len(MODELS))]
fig = plt.figure()
def plot_all(step, *args, **kwargs):
  plt.cla()
  plt.contour(*CONTOUR, 40, levels=LOSS.contour_levels)
  for i, model in enumerate(MODELS):
    x, y = model.layers[1].get_weights()[0][0]
    X, Y = TRAJECTORY[i]
    X[step], Y[step] = x, y
    for _ in range(SKIP):
        model.train_on_batch(FAKE_INPUT, FAKE_OUTPUT)
    plt.plot(X[:step+1], Y[:step+1], '-o', label=model.name, markevery=[-1])
  plt.plot(*LOSS.minimum_pos, 'o', label='Minimum')
  plt.grid()
  plt.legend()
  plt.xlim(LOSS.x_range)
  plt.ylim(LOSS.y_range)

anim = anm.FuncAnimation(fig, plot_all, interval=1, frames=FRAMES, init_func=lambda: None)

#anim.save("Sample.mp4", writer=anm.FFMpegFileWriter(fps=120))
plt.show()
