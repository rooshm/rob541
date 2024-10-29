import numpy as np
import sys
sys.path.append('../')
import geomotion.representationgroup as rgp
from vector import VectorBases
import numdifftools as ndt
from lie_algebra import GroupTangentVector

class MatrixVectorBases(VectorBases):
    def __init__(self, vectors: list, config = None, size = None) -> None:
        if config is not None:
            self.group = config.group
            self.config = config
        if isinstance(vectors, list):
            self.value = vectors
        elif isinstance(vectors, np.ndarray):
            self.value = []
            if config is not None:
                square_size = self.group.identity_element().rep.shape[0]
            elif size is not None:
                square_size = size[0]
            else:
                square_size = np.sqrt(vectors.shape[0])
            for i in range(vectors.shape[1]):
                self.value.append(vectors[:, i].reshape((square_size, square_size)))

    def flatten(self):
        return np.hstack([m.ravel(order='F').reshape(-1, 1) for m in self.value])

    def derep(self):
        return np.hstack([self.group.velocity_derep(self.config, v) for v in self.value])
    
    def __repr__(self) -> str:
        return f'MatrixVectorBases value {self.value}'


def matrix_derivative(func, config):
    reduced_func = lambda config: func(config).reshape(-1)
    jacobian = ndt.Jacobian(reduced_func)
    jacobian_matrix = jacobian(config)
    og = func(config)
    size = og.shape

    # print(f"jacobian_matrix: {jacobian_matrix} size: {size}")
    return MatrixVectorBases(vectors=jacobian_matrix, size=size)

class RepresentationGroup(rgp.RepresentationGroup):
    def __init__(self, representation_function_list, identity, derepresentation_function_list=None, specification_chart=0, normalization_function=None):
        super().__init__(representation_function_list, identity, derepresentation_function_list, specification_chart, normalization_function)

    def velocity_rep(self, g: rgp.RepresentationGroupElement, vector_coords: np.ndarray):
        m = matrix_derivative(self.representation_function_list[g.current_chart], g.value).value
        # print(f"m: {m} and vector_coords: {vector_coords.reshape(-1, 1)}")
        res = np.hstack([mi @ vector_coords.reshape(-1, 1) for mi in m])
        # print(f"res: {res}")
        return res

    def velocity_derep(self, g: rgp.RepresentationGroupElement, mat):

        single_mat = matrix_derivative(self.representation_function_list[g.current_chart], g.value).flatten()
        velocity = np.concat([mat[:, i].ravel() for i in range(mat.shape[1])])
        # print(f"single_mat: {single_mat} inv: {np.linalg.pinv(single_mat)} and velocity: {velocity}")
        return np.linalg.pinv(single_mat) @ velocity.reshape(-1, 1)

    def element(self,
                representation,
                initial_chart=0):
        """Instantiate a group element with a specified value"""
        g = RepresentationGroupElement(self,
                                       representation,
                                       initial_chart)
        return g

    def identity_element(self,
                         initial_chart=0):

        """Instantiate a group element at the identity"""
        g = RepresentationGroupElement(self,
                                       'identity',
                                       initial_chart)

        return g

    @property
    def representation_shape(self):
        return self.identity_rep.shape


class RepGroupTangentVector(GroupTangentVector):
    def __init__(self, group: RepresentationGroup, value: np.ndarray, config: rgp.RepresentationGroupElement) -> None:
        self.group = group
        self.config = config
        if len(value.shape) == 1:
            if value.shape[0] < group.identity_element().rep.ravel().shape[0]:
                self.value = group.velocity_rep(self.config, value)
            else:
                self.value = value.reshape(self.config.value.shape)
        else:
            self.value = value

    def flatten(self):
        return np.concat([self.value[:, i].ravel() for i in range(self.value.shape[1])])

    def derep(self):
        return self.group.velocity_derep(self.config, self.value).reshape(self.config.value.shape)
    
    def __repr__(self) -> str:
        return f'RepGroupTangentVector value {self.derep()} and configuration {self.config}'
    
class RepresentationGroupElement(rgp.RepresentationGroupElement):
    def __init__(self, group, representation, initial_chart=0):
        super().__init__(group, representation, initial_chart)

    def TL(self, gdot: RepGroupTangentVector):
        gcircright = self.rep @ gdot.value
        return RepGroupTangentVector(group=self.group, value=gcircright, config=self.L(gdot.config))
    
    def TR(self, gdot: RepGroupTangentVector):
        gcircleft = gdot.value @ self.rep
        return RepGroupTangentVector(group=self.group, value=gcircleft, config=self.R(gdot.config))
    
    def ad(self, gcircright: RepGroupTangentVector):
        """Lifted adjoint action"""
        g = self
        ginv = g.inverse
        # print(f"g: \n{g.rep} \nginv: \n{ginv.rep} \ngcircright: \n{gcircright.value}")
        return RepGroupTangentVector(group=self.group, value=g.rep @ gcircright.value @ ginv.rep, config=g)

    def ad_inv(self, gcircleft: RepGroupTangentVector):
        """Lifted adjoint inv action"""
        g = self
        ginv = g.inverse
        # print(f"g: \n{g.rep} \nginv: \n{ginv.rep} \ngcircleft: \n{gcircleft.value}")
        return RepGroupTangentVector(group=self.group, value=ginv.rep @ gcircleft.value @ g.rep, config=g)

def scale_shift_rep(g_value):
    g_rep = [[g_value[0], g_value[1]], [0, 1]]

    return g_rep

def scale_shift_derep(g_rep):
    g_value = [g_rep[0][0], g_rep[0][1]]

    return g_value

def scale_shift_normalization(g_rep):
    g_rep_normalized = [[g_rep[0][0], g_rep[0][1]], [0, 1]]

    return g_rep_normalized

RxRplus = RepresentationGroup(scale_shift_rep, [1, 0], scale_shift_derep, 0, scale_shift_normalization)

def SE2_rep(g_value):
    x = g_value[0]
    y = g_value[1]
    theta = g_value[2]

    g_rep = [[np.cos(theta), -np.sin(theta), x],
             [np.sin(theta), np.cos(theta), y],
             [0, 0, 1]]

    return g_rep


def SE2_derep(g_rep):
    x = g_rep[0][2]
    y = g_rep[1][2]
    theta = np.arctan2(g_rep[1][0], g_rep[0][0])

    g_value = [x, y, theta]
    return g_value

def SE2_normalize(g_rep):

    R = g_rep[0:2, 0:2]

    R_normalized = (1.5 * R) - (0.5 * np.matmul(np.matmul(R, np.transpose(R)), R))

    g_rep_normalized = np.concatenate([np.concatenate([R_normalized, g_rep[[0, 1], 2:]], 1), [[0, 0, 1]]])

    return(g_rep_normalized)

SE2 = RepresentationGroup(SE2_rep, [0, 0, 0], SE2_derep, 0, SE2_normalize)