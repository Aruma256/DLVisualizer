import numpy as np

class TestFunction:
    def __init__(self, f, init_pos, minimum_pos, x_range, y_range, contour_levels):
        self.f = f
        self.init_pos = init_pos
        self.minimum_pos = minimum_pos
        self.x_range = x_range
        self.y_range = y_range
        self.contour = self._get_contour(x_range, y_range)
        self.contour_levels = contour_levels
    def __call__(self, x, y):
        return self.f(x, y)
    def _get_contour(self, x_range, y_range):
        grid = [np.linspace(*x_range, 100),
                np.linspace(*y_range, 100)]
        x, y = np.meshgrid(*grid)
        z = self(x, y)
        return [x, y, z]


Beale = TestFunction(
    lambda x, y: (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2,
    init_pos=(1.5, 1.2),
    minimum_pos=(3, 0.5),
    x_range=(-4.5, 4.5),
    y_range=(-4.5, 4.5),
    contour_levels=np.linspace(0, 50, 20)
)
