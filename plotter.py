import matplotlib.pyplot as plt
import numpy as np
from S300_Construct_SE2 import RigidBody
from S300_Construct_SE2 import cornered_triangle
from geomotion import plottingfunctions as gplt

def add_marker(ax, loc, label, random_color=False):
    x, y, theta = loc
    if random_color:
        # random color from color map
        cmap = plt.get_cmap('viridis')
        color = cmap(np.random.rand())
    else:
        color = 'blue'
    ax.arrow(x, y, 0.1 * np.cos(theta), 0.1 * np.sin(theta), head_width=0.1, head_length=0.13, fc=color, ec=color)
    ax.text(x , y - 0.3, label, fontsize=12)

def add_velocity(ax, loc, vel, label = None, random_color=False):
    x, y, theta = loc
    x2, y2, theta2 = vel
    if random_color == True:
        cmap = plt.get_cmap('viridis')
        color = cmap(np.random.rand())
    else:
        color = 'black'

    ax.arrow(x, y, 0.5 * x2, 0.5 * y2, head_width=0.1, head_length=0.13, fc=color, ec=color, length_includes_head=True)
    ax.text(x , y - 0.3, label, fontsize=12)

def square_plt(a = -1.5, b = 3, title = 'Square Plot'):
    fig, ax = plt.subplots(layout='constrained') 
    ax.set_aspect('equal')
    fig.set_size_inches(6, 6)
    ax.grid(True)
    ax.plot([0, 0], [a, b], 'k')
    ax.plot([a, b], [0, 0], 'k')
    ax.set_title(title)
    return ax

def plot_vector_field(configs, values_list, title = 'Vector Field', padding = 1, color_idx = 0):
    fig, ax = plt.subplots(layout='constrained')
    ax.set_box_aspect(1)
    ax.set_aspect('equal')
    x_configs = configs[0, :]
    y_configs = configs[1, :]
    # draw axes to ensure full arrow is visible
    max_config_x = np.max(x_configs)
    min_config_x = np.min(x_configs)
    max_config_y = np.max(y_configs)
    min_config_y = np.min(y_configs)
    x_width = len(np.unique(x_configs))
    y_width = len(np.unique(y_configs))
    if len(x_configs) > 1 and len(y_configs) > 1:
        x_spacing = (max_config_x - min_config_x) / x_width
        y_spacing = (max_config_y - min_config_y) / y_width
    else:
        x_spacing = padding
        y_spacing = padding
    ax.plot([0, 0], [min(min_config_y - y_spacing, 0), max_config_y + y_spacing], '--k', alpha=0.2)
    ax.plot([min(min_config_x - x_spacing, 0), max_config_x + x_spacing], [0, 0], '--k', alpha=0.2)
    # plot vector field
    colors = ['red', 'black', 'blue', 'green', 'purple']
    for i, (values, color_str) in enumerate(zip(values_list, colors[color_idx:len(values_list) + color_idx])):
        q_artist = ax.quiver(x_configs, y_configs, 
                values[0, :], values[1, :], 
                angles='xy', scale_units='xy', scale = 2.7 * 1 / max(x_spacing, y_spacing), width = 0.004, color=color_str)
        ax.quiverkey(q_artist, 1.0 + (x_spacing / x_width), 1.0 - (y_spacing / y_width) * (i + 1), 1., label=f'Vector {i + 1}', labelpos='E', coordinates='axes')
    ax.set_title(title)
    fig.get_layout_engine().set(w_pad=x_spacing, h_pad=y_spacing, hspace=0, wspace=0)
    # fig.savefig(f'{title}.png')

def plot_bodies(ax, tvs, color='red'):
    spot_color = gplt.crimson
    for i, tv in enumerate(tvs):
        body = RigidBody(cornered_triangle(.25, spot_color), tv.config)
        body.draw(ax)
        add_velocity(ax, tv.config, tv.derep())

