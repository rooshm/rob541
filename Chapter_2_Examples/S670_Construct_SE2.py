import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)
from geomotion import representationliegroup as rlgp
from geomotion import group as gp
from geomotion import utilityfunctions as ut
import numpy as np
from matplotlib import pyplot as plt


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

    return (g_rep_normalized)


SE2 = rlgp.RepresentationLieGroup(SE2_rep, [0, 0, 0], SE2_derep, 0, SE2_normalize)


class RigidBodyPlotInfo:

    def __init__(self, **kwargs):

        if 'plot_locus' in kwargs:
            self.plot_locus = kwargs['plot_locus']

        if 'plot_style' in kwargs:
            self.plot_style = kwargs['plot_style']


def cornered_triangle(r, spot_color, **kwargs):

    def T1(body):
        return SE2.element_set(ut.GridArray([[r, 0, 0],
                                       [r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), 0],
                                       [r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), 0]], 1),
                         0, "element")

    def T2(*args, **kwargs):
           return SE2.element_set(ut.GridArray([[r, 0, 0],
                                       [r / 3 * np.cos(2 * np.pi / 3) + (2 * r / 3), r / 3 * np.sin(2 * np.pi / 3), 0],
                                       [r / 3 * np.cos(4 * np.pi / 3) + (2 * r / 3), r / 3 * np.sin(4 * np.pi / 3), 0]],
                                      1),
                         0, "element")

    plot_locus = [T1, T2]

    plot_style = [{"edgecolor": 'black', "facecolor": 'white'} | kwargs,
                  {"edgecolor": 'black', "facecolor": spot_color} | kwargs]

    plot_info = RigidBodyPlotInfo(plot_locus=plot_locus, plot_style=plot_style)

    return plot_info


class RigidBody:

    def __init__(self,
                 plot_info,
                 position=SE2.identity_element()):
        self.plot_info = plot_info
        self.position = position

    def draw(self,
             axis):
        plot_locus = self.plot_info.plot_locus
        plot_options = self.plot_info.plot_style

        for i, p in enumerate(plot_locus):
            # Transform the locally expressed positions of the drawing points by the position of the body
            plot_points_global = self.position * p(self)
            plot_points_global_grid = plot_points_global.grid

            axis.fill(plot_points_global_grid[0], plot_points_global_grid[1], 'black', **(plot_options[i]))
            #print(plot_points_global_grid[0], "\n", plot_points_global_grid[1])
