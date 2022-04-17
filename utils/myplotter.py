import matplotlib.pyplot as plt

def plot_data(data, label):
    plt.plot(data, label=label, marker="o", linestyle="-")

def setup_plot(figure_title, y_axis_label):
    plt.title(figure_title)
    plt.ylabel(y_axis_label)  # y label
    plt.xlabel("Iteration Rounds")  # x label

def show():
    plt.legend()
    plt.show()