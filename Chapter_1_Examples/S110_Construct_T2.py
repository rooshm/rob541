import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)
#! /usr/bin/python3
import numpy as np
from geomotion import manifold as md
from geomotion import utilityfunctions as ut


# Torus. Charts are chosen such that in a "doughnut" embedding:
# First chart's first axis is along the major circumference and second axis is along the minor circumference,
#
# Second chart also has first axis along major circumference, and is centered a quarter revolution away from the center
#   of the first chart along both the major and minor axes
#
# Third chart is centered opposite both of the first two charts, aligned with first axis on the major axis

# Define tranition maps between the three charts
def first_to_second(input_coords):
    output_coords = np.empty_like(input_coords)
    output_coords[0] = ut.cmod(input_coords[0] + 0.5, 1)
    output_coords[1] = ut.cmod(input_coords[1] + 0.5, 1)

    return output_coords


def first_to_third(input_coords):
    output_coords = np.empty_like(input_coords)
    output_coords[0] = ut.cmod(input_coords[0] + 0.25, 1)
    output_coords[1] = ut.cmod(input_coords[1] + 0.25, 1)

    return output_coords


def second_to_first(input_coords):
    output_coords = np.empty_like(input_coords)
    output_coords[0] = ut.cmod(input_coords[0] - 0.5, 1)
    output_coords[1] = ut.cmod(input_coords[1] - 0.5, 1)

    return output_coords


def second_to_third(input_coords):
    output_coords = np.empty_like(input_coords)
    output_coords[0] = ut.cmod(input_coords[0] - 0.25, 1)
    output_coords[1] = ut.cmod(input_coords[1] - 0.25, 1)

    return output_coords


def third_to_first(input_coords):
    output_coords = np.empty_like(input_coords)
    output_coords[0] = ut.cmod(input_coords[0] - 0.25, 1)
    output_coords[1] = ut.cmod(input_coords[1] - 0.25, 1)

    return output_coords


def third_to_second(input_coords):
    output_coords = np.empty_like(input_coords)
    output_coords[0] = ut.cmod(input_coords[0] + 0.25, 1)
    output_coords[1] = ut.cmod(input_coords[1] + 0.25, 1)

    return output_coords


# Construct transition table
transition_table = [[None, first_to_second, first_to_third],
                    [second_to_first, None, second_to_third],
                    [third_to_first, third_to_second, None]]

# Generate the manifold
T2 = md.Manifold(transition_table, 2)
