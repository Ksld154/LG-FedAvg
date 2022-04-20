import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

def plot_data(data, label):
    plt.plot(data, label=label, marker="o", linestyle="-")

def setup_plot(figure_title, y_axis_label, figure_idx):
    ax1 = plt.figure(figure_idx).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis
    plt.title(figure_title)
    plt.ylabel(y_axis_label)  # y label
    plt.xlabel("Iteration Rounds")  # x label

def legend():
    plt.legend()

def save_figure(base_dir, filename):
    image_path = os.path.join(base_dir, filename)
    plt.savefig(image_path)


def show():
    plt.legend()
    plt.show()
