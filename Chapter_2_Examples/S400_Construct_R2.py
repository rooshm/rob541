import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)
from geomotion import diffmanifold as dm
import numpy as np


def polar_to_cartesian(polar_coords):
    cartesian_coords = np.empty_like(polar_coords)
    cartesian_coords[0] = polar_coords[0] * np.cos(polar_coords[1])
    cartesian_coords[1] = polar_coords[0] * np.sin(polar_coords[1])

    return cartesian_coords


def cartesian_to_polar(cartesian_coords):
    polar_coords = np.empty_like(cartesian_coords)
    polar_coords[0] = np.sqrt((cartesian_coords[0] * cartesian_coords[0]) + (cartesian_coords[1] * cartesian_coords[1]))
    polar_coords[1] = np.arctan2(cartesian_coords[1], cartesian_coords[0])

    return polar_coords


transition_table = [[None, cartesian_to_polar], [polar_to_cartesian, None]]

R2 = dm.DiffManifold(transition_table, 2)
