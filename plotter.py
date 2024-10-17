import matplotlib.pyplot as plt
import numpy as np

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

def square_plt():
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    fig.set_size_inches(6, 6)
    ax.grid(True)
    ax.plot([0, 0], [-1.5, 3], 'k')
    ax.plot([-1.5, 3], [0, 0], 'k')
    return ax