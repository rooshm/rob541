from enum import Enum
import numpy as np

def affine_repr(elems):
    mat = np.eye(len(elems) + 1)
    for i, elem in enumerate(elems):
        mat[i, -1] = elem
    return mat

class OpEnum(Enum):
    """
    Convenience enum to access common groups
    """
    SCALAR_ADD = 1
    SCALAR_PRODUCT = 2
    MODULAR_ADD = 3
    AFFINE_ADD = 4
    ELEM_MULTIPLY = 5
    SO2 = 5

class OpGen(object):
    """Convenience class to access combination of operation, inverse, identities etc for groups"""
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
            self.representation = affine_repr
            self.derepresentation = lambda x: x[:-1, -1]

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