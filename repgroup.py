#!/usr/bin/env python3
import numpy as np
from group import Group, GroupElement
from convenience import OpEnum, OpGen
import matplotlib.pyplot as plt
from plotter import add_marker

np.set_printoptions(precision=3)

class RepGroup(Group):
    def __init__(self, representation, derepresentation, identity, params={}):
        operation = lambda x, y: x @ y
        self.representation = representation
        self.derepresentation_fn = derepresentation
        if isinstance(identity, np.ndarray):
            self.identity = identity
        else:
            self.identity = RepGroup.make_identity(identity, params)
        inverse = lambda x: np.linalg.inv(x)
        super().__init__(operation, inverse, identity, params)
    
    def make_identity(identity, params):
        return np.eye(params['dim'] + 1)
    
    def element(self, value):
        return RepGroupElement(self, value)
    
    def identity_element(self):
        return self.element(self.identity)


class RepGroupElement(GroupElement):
    def __init__(self, group, value):
        if isinstance(value, np.ndarray) and value.shape == group.identity.shape:
            save_value = value
        else:
            save_value = group.representation(value)
        super().__init__(group, save_value)
    
    def left_action(self, e):
        result = self.group.operation(self.value, e.value)
        return self.group.element(result)
    
    def right_action(self, e):
        result = self.group.operation(e.value, self.value)
        return self.group.element(result)
    
    @property
    def derepresentation(self):
        return self.group.derepresentation_fn(self.value)

class SE2(RepGroup):
    def __init__(self):
        super().__init__(SE2.se2_repr, SE2.se2_derepr, np.eye(3))

    @staticmethod
    def se2_repr(value):
        x, y, theta = value
        return np.array([[np.cos(theta), -np.sin(theta), x],
                        [np.sin(theta), np.cos(theta), y],
                        [0, 0, 1]])

    @staticmethod
    def se2_derepr(mat):
        return np.array([mat[0, 2], mat[1, 2], np.arctan2(mat[1, 0], mat[0, 0])]).reshape(3)


  