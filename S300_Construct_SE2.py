import sys
sys.path.append('../')
from rep_lie_algebra import SE2
from geomotion import group as gp
from geomotion import utilityfunctions as ut
import numpy as np

class RigidBodyPlotInfo:

    def __init__(self, **kwargs):

        if 'plot_points' in kwargs:
            self.plot_points = kwargs['plot_points']

        if 'plot_style' in kwargs:
            self.plot_style = kwargs['plot_style']


def cornered_triangle(r, spot_color, **kwargs):
    T1 = SE2.element_set(ut.GridArray([[r, 0, 0],
                                       [r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), 0],
                                       [r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), 0]], 1),
                         0, "element")

    T2 = SE2.element_set(ut.GridArray([[r, 0, 0],
                                       [r / 3 * np.cos(2 * np.pi / 3) + (2 * r / 3), r / 3 * np.sin(2 * np.pi / 3), 0],
                                       [r / 3 * np.cos(4 * np.pi / 3) + (2 * r / 3), r / 3 * np.sin(4 * np.pi / 3), 0]],
                                      1),
                         0, "element")

    plot_points = [T1, T2]

    plot_style = [{"edgecolor": 'black', "facecolor": 'white'} | kwargs,
                  {"edgecolor": 'black', "facecolor": spot_color} | kwargs]

    plot_info = RigidBodyPlotInfo(plot_points=plot_points, plot_style=plot_style)

    return plot_info


class RigidBody:

    def __init__(self,
                 plot_info,
                 position=SE2.identity_element()):
        self.plot_info = plot_info
        self.position = position

    def draw(self,
             axis):
        plot_points = self.plot_info.plot_points
        plot_options = self.plot_info.plot_style

        for i, p in enumerate(plot_points):
            # Transform the locally expressed positions of the drawing points by the position of the body
            plot_points_global = self.position * p
            plot_points_global_grid = plot_points_global.grid

            axis.fill(plot_points_global_grid[0], plot_points_global_grid[1], 'black', **(plot_options[i]))
            #print(plot_points_global_grid[0], "\n", plot_points_global_grid[1])
