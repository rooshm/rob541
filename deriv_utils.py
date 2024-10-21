from vector import TangentVector, GroupTangentVector
from repgroup import RepGroupElement
import numpy as np
import numdifftools as nd

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

def scaled_group_left_action(delta: float, a: RepGroupElement, b: RepGroupElement) -> np.ndarray:
    """ Compute the scaled group action of a group element (delta * a) o b"""
    if isinstance(delta, np.ndarray):
        delta = delta.flatten()[0]
    scaled_group_element = RepGroupElement(value=delta * a.value, group=a.group)
    return scaled_group_element.left_action(b).value

def derivative_group_left_action(direction: RepGroupElement, config: RepGroupElement) -> np.ndarray:
    reduced_group_action = lambda delta: scaled_group_left_action(delta, direction, config)
    deriv = nd.Derivative(reduced_group_action)(0.)
    return GroupTangentVector(val=RepGroupElement(value=deriv, group=direction.group), config=config)

def scaled_group_right_action(delta: float, a: RepGroupElement, b: RepGroupElement) -> GroupTangentVector:
    """ Compute the scaled group action of a group element b o (delta * a)"""
    if isinstance(delta, np.ndarray):
        delta = delta.flatten()[0]
    scaled_group_element = RepGroupElement(value=delta * a.value, group=a.group)
    return scaled_group_element.right_action(b).value

def derivative_group_action_bases(direction: RepGroupElement, config: RepGroupElement, action = "left"):
    def func(direction, config, i, action, delta):
        if isinstance(delta, np.ndarray):
            delta = delta.flatten()[0]
        flattened_direction = direction.derepresentation.flatten()
        flattened_direction[i] = flattened_direction[i] + delta
        new_basis = RepGroupElement(value=flattened_direction, group=direction.group)
        if action == "left":
            return new_basis.left_action(config).value
        elif action == "right":
            return new_basis.right_action(config).value
        else:
            raise ValueError("Invalid action")
    
    gtvs = []
    flattened_direction = direction.derepresentation.flatten()
    for i in range(flattened_direction.shape[0]):
        reduced_group_action = lambda delta: func(direction, config, i, action, delta)
        deriv = nd.Derivative(reduced_group_action)(0.)
        # print(f'deriv: {deriv} at config {config} for basis g{i}')
        gtv = GroupTangentVector(val=RepGroupElement(value=deriv, group=direction.group), config=config)
        gtvs.append(gtv)
    return gtvs

