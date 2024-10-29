from vector import TangentVector
from repgroup import RepGroupElement
import numpy as np
import numdifftools as nd
from lie_algebra import GroupTangentVector

# example fn to model real functions off of
def _fn(config: np.ndarray, delta: float) -> np.ndarray:
    """
    Function to calculate the value of a function at a configuration
    If delta is 0, the output is the configuration
    """
    if isinstance(delta, np.ndarray):
        delta = delta.flatten()(0)
    if np.isclose(delta, 0.):
        return config
    return delta * config

def derivative_in_direction(func, config) -> TangentVector:
    """ Compute the derivative in direction of function at a given point

    :param func: func of type fn, the function to calculate the derivative of wrt delta at the config
    :param config: config to calculate the derivative at
    :returns TangentVector: location is at config
    """
    reduced_func = lambda delta : func(config, delta).reshape(config.shape)
    deriv = nd.Derivative(reduced_func)(0.)
    return TangentVector(value=deriv, config=config)

def compute_jacobian(func, config) -> TangentVector:
    """ Compute the jacobian of the function at the configuration

    :param func: func of type fn, the function to calculate the derivative of wrt delta at the config
    :param config: config to calculate the derivative at
    """
    reduced_func = lambda delta: func(config, delta).reshape(config.shape)
    jacobian = nd.Jacobian(reduced_func)
    jacobian_matrix = jacobian(0.)
    return TangentVector(value=jacobian_matrix, config=config)


