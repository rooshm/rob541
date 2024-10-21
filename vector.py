import numpy as np
from repgroup import RepGroup, RepGroupElement

class TangentVector(object):
    def __init__(self, value, config) -> None:
        self.value = value
        self.config = config

    def __repr__(self) -> str:
        return f'TangentVector(value: {self.value}, config: {self.config})'

    def __add__(self, other):
        if not isinstance(other, TangentVector):
            raise ValueError('Cannot add a tangent vector with a non-tangent vector')
        if self.config != other.config:
            raise ValueError('Cannot add two tangent vectors with different configurations')
        return TangentVector(self.value + other.value, self.config)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return TangentVector(other * self.value, self.config)
        if isinstance(other, np.ndarray):
            # check if square matrix
            if other.shape[0] != other.shape[1]:
                raise ValueError('Cannot multiply a tangent vector with a non-square matrix')
            if other.shape[1] != self.value.shape[0]:
                raise ValueError('Cannot multiply a tangent vector with a matrix of different size')
            return TangentVector(other @ self.value, self.config)
        raise ValueError('Unsupported operand type')

    def __mul__(self, other):
        if isinstance(other, (int, float)): # only scalar multiplication
            return TangentVector(self.value * other, self.config)
        raise ValueError('Only scalar multiplication supported for left multiplication')

class VectorBases(object):
    def __init__(self, vectors: list, inverse = None) -> None:
        self.vectors = vectors
        self.inverse = inverse

    def __repr__(self) -> str:
        return f'VectorBases(vectors: {self.vectors})'

    def flatten(self):
        return np.hstack([vector.value.reshape(-1,1) for vector in self.vectors])

    def configs(self):
        return np.hstack([vector.config.reshape(-1,1) for vector in self.vectors])

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

class GroupTangentVector(object):
    def __init__(self, val: RepGroupElement, config: RepGroupElement) -> None:
        self.config = config
        self.group = self.config.group
        self.value = val
    
    def as_tv(self):
        return TangentVector(value=self.value.derepresentation, config=self.config.derepresentation)

    def __repr__(self) -> str:
        return f'GroupTangentVector(value: {self.value}, config: {self.config}, group: {self.group})'
    
    def __add__(self, other):
        raise NotImplementedError('Prefer left and right composition')

    def __mul__(self, other):
        raise NotImplementedError('Prefer left and right composition')
    
    def __rmul__(self, other):
        raise NotImplementedError('Prefer left and right composition')

