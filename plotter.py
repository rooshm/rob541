import matplotlib.pyplot as plt
import numpy as np

def add_marker(ax, loc, label):
    x, y, theta = loc
    ax.arrow(x, y, 0.1 * np.cos(theta), 0.1 * np.sin(theta), head_width=0.07, head_length=0.1)
    ax.text(x - 0.2, y - 0.2, label, fontsize=12)
