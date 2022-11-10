import numpy as np


class Line:
    def __init__(self, p1: tuple, p2: tuple):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.norm = np.linalg.norm(self.p1 - self.p2)
        if p2[0] == p1[0]:
            self.dir = float("inf")
            self.x_pos = p1[0]
        else:
            self.dir = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.const_term = self.p1[1] - self.dir * self.p1[0]

    def calculate_y(self, x0: float) -> float:
        """
        Get y value from line equation and x0
        :param x0: point at which the y should be calculated
        :return: value of y
        """
        return self.dir * x0 + self.const_term  # y=ax+b -> x = (y-b)/a

    def calculate_x(self, y0: float) -> float:
        """
        Get y value from line equation and y0
        :param y0: point at which the x should be calculated
        :return: value of x
        """
        return (y0 - self.const_term) / self.dir
