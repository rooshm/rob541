import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)
#! /usr/bin/python3
import numpy as np
from geomotion import representationgroup as rgp

np.set_printoptions(precision=3, floatmode='maxprec')  # Make things print nicely

# """ Scalar addition """
#
#
# def scalar_addition_rep(g_value):
#     g_rep = [[1, g_value[0]], [0, 1]]
#     return g_rep
#
#
# def scalar_addition_derep(g_rep):
#     g_value = g_rep[0][1]
#     return g_value
#
#
# R1plus = rgp.RepresentationGroup(scalar_addition_rep, 0, scalar_addition_derep)
#
# g1 = R1plus.element(2)
# g2 = R1plus.element(3)
#
# g3 = g1 * g2
#
# print("Addition-group composition of ", g1, " and ", g2, " is ", g3)
#
# """ Scalar addition """
#
#
# def scalar_multiplication_rep(g_value):
#     g_rep = [[g_value[0]]]
#     return g_rep
#
#
# def scalar_multiplication_derep(g_rep):
#     g_value = g_rep[0][0]
#     return g_value
#
#
# R1times = rgp.RepresentationGroup(scalar_multiplication_rep, 1, scalar_multiplication_derep)
#
# g1 = R1times.element(2.)
# g2 = R1times.element(3.)
#
# g3 = g1 * g2
#
# print("Multiplication-group composition of ", g1, " and ", g2, " is ", g3)


def modular_addition_rep_a(g_value):
    #print("rep_a")
    g_radians = 2 * np.pi * g_value[0]
    g_rep = [[np.cos(g_radians), -np.sin(g_radians)],
             [np.sin(g_radians), np.cos(g_radians)]]
    return g_rep


def modular_addition_rep_b(g_value):
    #print("rep_b")
    g_radians = 2 * np.pi * g_value[0]
    g_rep = [[np.cos(g_radians), -np.sin(g_radians)],
             [np.sin(g_radians), np.cos(g_radians)]]
    return g_rep


def modular_addition_derep_a(g_rep):
    #print("derep_a")
    g_value = np.arctan2(g_rep[1][0], g_rep[0][0]) / (2 * np.pi)

    #print("derep_a in function is ", g_value, " and is of type ", type(g_value))
    return g_value


def modular_addition_derep_b(g_rep):
    #print("derep_b")
    g_value = np.arctan2(g_rep[1][0], g_rep[0][0]) / (2 * np.pi)
    if g_value < 0:
        g_value += 1

    return g_value


rep_list = [modular_addition_rep_a, modular_addition_rep_b]
derep_list = [modular_addition_derep_a, modular_addition_derep_b]

S1plus = rgp.RepresentationGroup(rep_list, 0, derep_list)

g1 = S1plus.element(.25)
g2 = S1plus.element(.875)

g3 = g1 * g2

print("Modular-addition-group composition of  ", g1, " and ", g2, " is ", g3)

g1a = g1.transition(1)
g2a = g2.transition(1)
g3a = g3.transition(1)

print(g2.current_chart, g2a.current_chart)

print("Modular-addition-group composition of  ", g1a, " and ", g2a, " is ", g3a)
