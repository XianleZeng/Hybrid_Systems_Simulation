import numpy as np
import matplotlib.pyplot as plt
import scipy
from absl import logging
logging.set_verbosity(logging.DEBUG)


class BouncingBall:
    def __init__(self, lambda_=0.8):
        self.mass = 1.0
        self.gravity = 9.8
        self.lambda_ = lambda_

    def flow_map(self, state, u):
        x1, x2 = state

        x_dot = np.zeros(2)
        x_dot[0] = x2
        x_dot[1] = -self.gravity

        return x_dot

    def jump_map(self, state):
        x1, x2 = state

        x_plus = np.zeros(2)
        x_plus[0] = x1
        x_plus[1] = -self.lambda_ * x2
        return x_plus

    def collision_witness(self, state):
        x1, x2 = state

        return x1


class CompassGait:
    def __init__(self, gamma=0.05):
        self.leg_length = 1.0
        self.mass_leg = 5.0
        self.mass_hip = 10.0
        self.gravity = 9.81
        self.gamma = gamma

    def flow_map(self, state, u):
        x1, x2, x3, x4 = state
        l = self.leg_length
        a = l / 2
        b = l / 2
        m = self.mass_leg
        m_h = self.mass_hip
        g = self.gravity

        B = np.array([0, 0, 1, -1])

        x_dot = np.zeros(4)
        x_dot[0] = x3
        x_dot[1] = x4
        x_dot[2] = l * (b * l * m * x3 ** 2 * np.sin(x2 - x1) - g * (-a * m - l * m - l * m_h) * np.sin(x2)) * np.cos(
            x2 - x1) / (a ** 2 * b * m - b * l ** 2 * m * np.cos(x2 - x1) ** 2 + b * l ** 2 * m + b * l ** 2 * m_h) + (
                               -b * g * m * np.sin(x1) - b * l * m * x4 ** 2 * np.sin(x2 - x1)) * (
                               a ** 2 * m + l ** 2 * m + l ** 2 * m_h) / (
                               a ** 2 * b ** 2 * m ** 2 - b ** 2 * l ** 2 * m ** 2 * np.cos(
                           x2 - x1) ** 2 + b ** 2 * l ** 2 * m ** 2 + b ** 2 * l ** 2 * m * m_h)
        x_dot[3] = l * (-b * g * m * np.sin(x1) - b * l * m * x4 ** 2 * np.sin(x2 - x1)) * np.cos(x2 - x1) / (
                    a ** 2 * b * m - b * l ** 2 * m * np.cos(x2 - x1) ** 2 + b * l ** 2 * m + b * l ** 2 * m_h) + (
                               b * l * m * x3 ** 2 * np.sin(x2 - x1) - g * (-a * m - l * m - l * m_h) * np.sin(x2)) / (
                               a ** 2 * m - l ** 2 * m * np.cos(x2 - x1) ** 2 + l ** 2 * m + l ** 2 * m_h)

        x_dot = x_dot + B * u

        return x_dot

    def jump_map(self, state):
        x1, x2, x3, x4 = state
        l = self.leg_length
        a = l / 2
        b = l / 2
        m = self.mass_leg
        m_h = self.mass_hip
        g = self.gravity
        w4 = -a * b * m * x3 * np.cos(x2 - x1) / (
                    a ** 2 * b * m ** 2 + b * l ** 2 * m * m_h - b * l * m * np.cos(x2 - x1) ** 2 + b * l * m) + x4 * (
                         -a * b * m * (a ** 2 * m - b * np.cos(x2 - x1) + l ** 2 * m_h + l) / (
                             a ** 2 * b ** 2 * m ** 2 + b ** 2 * l ** 2 * m * m_h - b ** 2 * l * m * np.cos(
                         x2 - x1) ** 2 + b ** 2 * l * m) + (
                                     -a * b * m + (2 * a * l * m + l ** 2 * m_h) * np.cos(x2 - x1)) * np.cos(
                     x2 - x1) / (a ** 2 * b * m ** 2 + b * l ** 2 * m * m_h - b * l * m * np.cos(
                     x2 - x1) ** 2 + b * l * m))
        w3 = -a * b * m * x3 / (
                    a ** 2 * l * m ** 2 + l ** 3 * m * m_h - l ** 2 * m * np.cos(x2 - x1) ** 2 + l ** 2 * m) + x4 * (
                         -a * b * m * (-b + l * np.cos(x2 - x1)) / (
                             a ** 2 * b * l * m ** 2 + b * l ** 3 * m * m_h - b * l ** 2 * m * np.cos(
                         x2 - x1) ** 2 + b * l ** 2 * m) + (
                                     -a * b * m + (2 * a * l * m + l ** 2 * m_h) * np.cos(x2 - x1)) / (
                                     a ** 2 * l * m ** 2 + l ** 3 * m * m_h - l ** 2 * m * np.cos(
                                 x2 - x1) ** 2 + l ** 2 * m))
        return np.array([x2, x1, w3, -w4])

    def collision_witness(self, state):
        x1, x2, _, _ = state
        gamma = self.gamma
        temp = 2 * gamma + x1 + x2
        return min(temp, x1 - x2)