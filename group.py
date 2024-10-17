#!/bin/env python
from enum import Enum

# from logdecorator import log_on_start, log_on_end, log_on_error
import numpy as np
import matplotlib.pyplot as plt
from plotter import add_marker
from convenience import OpEnum, OpGen
np.set_printoptions(precision=3)

class GroupElement(object):
    def __init__(self, group, value):
        self.group = group
        self.value = value

    def left_action(self, e):
        result = self.group.operation(self.value, e.value)
        return self.group.element(result)

    def right_action(self, e):
        result = self.group.operation(e.value, self.value)
        return self.group.element(result)
    
    @property
    def inverted_element(self):
        return self.group.element(self.group.inverse(self.value))

    def AD(self, e):
        return self.left_action(e).left_action(self.inverted_element)
    
    def AD_inv(self, e):
        return self.inverted_element.left_action(e).left_action(self)

class Group(object):
    def __init__(self, operation, inverse, identity, params={}):
        self.identity = identity
        self.operation = operation
        self.inverse = inverse
        self.params = params

    def element(self, value):
        return GroupElement(self, value)
    
    def identity_element(self):
        return self.element(self.identity)


class DirectProduct(Group):
    def __init__(self, group1, group2):
        self.operation = lambda x, y: [group1.operation(x[0], y[0]), group2.operation(x[1], y[1])]
        self.inverse = lambda x, y: [group1.operation(group1.inverse(x[0]), group1.inverse(y[0])),
                                      group2.operation(group2.inverse(x[1], group2.inverse(y[1])))]
        self.identity = [group1.identity, group2.identity]
        self.params = {**group1.params, **group2.params}
        super().__init__(self.operation, self.inverse, self.identity, self.params)

class SemiDirectProduct(Group):
    def __init__(self, group1, group2):
        self.operation = lambda x, y: [group1.operation(x[0], y[0]), group2.operation(group1.operation(x[0], y[1]), x[1])]
        self.inverse = lambda x, y: SemiDirectProduct._inv(group1, group2, x, y)
        self.identity = [group1.identity, group2.identity]
        self.params = {**group1.params, **group2.params}
        super().__init__(self.operation, self.inverse, self.identity, self.params)
    
    @staticmethod
    def _inv(group1, group2, a, b):
        x, y = a[0], a[1]
        u, v = b[0], b[1]
        m = group1.operation(group1.inverse(x), group1.inverse(u))
        minusxv = group2.inverse(group1.operation(x, v))
        n = group1.operation(m, group2.operation(minusxv, group1.inverse(y)))
        return [m, n]


class SO2(Group):
    def __init__(self):
        op = OpGen(OpEnum.SO2)
        super().__init__(op.operation, op.inverse, op.operation, op.params)
    
    @staticmethod
    def Rot2D(theta):
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    @staticmethod
    def Theta(mat):
        return np.arctan2(mat[1, 0], mat[0, 0])
    
    def element(self, value):
        if isinstance(value, np.ndarray):
            return GroupElement(self, value)
        else:
            return GroupElement(self, SO2.Rot2D(value))

class SE2(Group):
    def __init__(self):
        self.operation = lambda a, b: SE2.xytheta(SE2.Mat2D(a) @ SE2.Mat2D(b))
        self.identity = np.eye(3)
        self.inverse = lambda x: SE2.xytheta(np.linalg.inv(SE2.Mat2D(x)))
        super().__init__(self.operation, self.inverse, self.identity)
    
    @staticmethod
    def Mat2D(value):
        x, y, theta = value
        mat = np.eye(3)
        mat[:2, :2] = SO2.Rot2D(theta)
        mat[:2, 2] = [x, y]
        return mat
    
    @staticmethod
    def xytheta(mat):
        return np.array([mat[0, 2], mat[1, 2], SO2.Theta(mat[:2, :2])]).reshape(3)