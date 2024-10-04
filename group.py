#!/bin/env python
from enum import Enum
# from logdecorator import log_on_start, log_on_end, log_on_error
import numpy as np
import matplotlib.pyplot as plt
from plotter import add_marker

class OpEnum(Enum):
    """
    Convenience class to access common groups
    """
    SCALAR_ADD = 1
    SCALAR_PRODUCT = 2
    MODULAR_ADD = 3
    AFFINE_ADD = 4
    ELEM_MULTIPLY = 5
    SO2 = 5

class OpGen(object):
    """Generate operation, inverse, identities for OpEnum provided groups"""
    def __init__(self, op: OpEnum, params={}):
        if op == OpEnum.SCALAR_ADD:
            self.params = params
            self.operation = lambda x, y: x + y
            self.inverse = lambda x: -x
            self.identity = 0
        
        elif op == OpEnum.AFFINE_ADD:
            self.params = params
            if not self.params.get('dim'):
                raise ValueError('Dimension must be specified for affine addition')
            self.operation = lambda x, y: np.asarray(x) + np.asarray(y)
            self.inverse = lambda x: -np.asarray(x)
            self.identity = np.zeros(self.params['dim'])

        elif op == OpEnum.MODULAR_ADD:
            self.params = params
            if not self.params.get('phi'):
                raise ValueError('Modulus must be specified for modular addition')
            self.operation = lambda x, y: (x + y) % self.params['phi']
            self.inverse = lambda x: x
            self.identity = 0

        elif op == OpEnum.SCALAR_PRODUCT:
            self.params = params
            self.operation = lambda x, y: x * y
            self.inverse = lambda x: 1/x
            self.identity = 1

        elif op == OpEnum.ELEM_MULTIPLY:
            self.params = params
            if not self.params.get('dim'):
                raise ValueError('Dimension must be specified for elementwise multiplication')
            self.operation = lambda x, y: np.asarray(x) * np.asarray(y)
            self.inverse = lambda x: np.reciprocal(x)
            self.identity = np.ones(self.params['dim'])
        
        elif op == OpEnum.SO2:
            self.params = params
            self.operation = lambda x, y: x @ y
            self.identity = np.array([0., 1.], [1., 0.])
            self.inverse = lambda x: np.inv(x)

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
        return np.arccos(mat[0, 0])
    
    def element(self, value):
        if isinstance(value, np.ndarray):
            return GroupElement(self, value)
        else:
            return GroupElement(self, SO2.Rot2D(value))

class SE2(Group):
    def __init__(self):
        self.operation = lambda a, b: a @ b
        self.identity = np.eye(3)
        self.inverse = lambda x: np.linalg.inv(x)
        super().__init__(self.operation, self.inverse, self.identity)
    
    def element(self, value):
        if isinstance(value, np.ndarray):
            return GroupElement(self, value)
        else:
            x, y, theta = value[0], value[1], value[2]
            mat = np.eye(3)
            mat[:2, :2] = SO2.Rot2D(theta)
            mat[:2, 2] = [x, y]
        return GroupElement(self, mat)
    
    @staticmethod
    def xytheta(mat):
        return np.array([mat[0, 2], mat[1, 2], SO2.Theta(mat[:2, :2])]).reshape(3)

if __name__ == '__main__':
    # deliverable 1
    se2 = SE2()
    g = se2.element([2, 1, np.pi/2])
    h = se2.element([0, -1, -np.pi/4])
    gh = g.left_action(h).value
    hg = h.left_action(g).value
    print(se2.xytheta(gh))
    print(se2.xytheta(hg))

    # deliverable 2
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    ax.set_xlim(-3, 4)
    ax.set_ylim(-3, 4)
    ax.grid(True)
    # draw the axes
    ax.plot([0, 0], [-3, 4], 'k')
    ax.plot([-3, 4], [0, 0], 'k')
    add_marker(ax, se2.xytheta(gh), "gh")
    add_marker(ax, se2.xytheta(hg), "hg")
    add_marker(ax, se2.xytheta(g.value), "g")
    add_marker(ax, se2.xytheta(h.value), "h")
    plt.show()
    

    



