import argparse
import sys
import os
import datetime

import utils.csv_exporter
import utils.myplotter
import utils.eps_plotter
import utils.group_plotter

from constants import *

from matplotlib import markers
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

import numpy as np

from constants import *

DEFAULT_FIGURE_SIZE = 2
color_options  = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
linestyle_options = ['-', '--', '-.', ':']
marker_options = ['o', '*', '.', ',', 'x', 'P', 'D', 'H']
hatch_options = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

MODEL_NAME = ['Baseline', 'Layer Freezing w/ PCME']

def set_figure_size(figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(4*DEFAULT_FIGURE_SIZE, 3*DEFAULT_FIGURE_SIZE)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

def legend():
    plt.legend(loc='lower right', frameon=False)

def save_figure(base_dir, filename):
    image_path = os.path.join(base_dir, filename)
    print(image_path)
    plt.savefig(image_path)

def show():
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def block_show():
    plt.show()
    plt.close()



def opt_parser():
    usage = 'Merge FL training results and plot figure from static freezing results and Gradually Freezing results.'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--load_file', type=str, default='', help='Load Static Freeze results from path')
    
    return parser.parse_args()


def get_merged_data(filepath):
    data = utils.csv_exporter.import_csv(filepath=filepath)
    print(len(data))

    freq_1 = []
    freq_5 = []
    for row in data:
        if int(row['freq']) == 1:
            freq_1.append(row)
        elif int(row['freq']) == 5:
            freq_5.append(row)
    # print(len(freq_1), len(freq_5))
    return [freq_1, freq_5]


def plot_duration_group(all_data, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(4*DEFAULT_FIGURE_SIZE, 3*DEFAULT_FIGURE_SIZE)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    all_group_data = []
    for group_data in all_data:
        t = []
        for idx, model in enumerate(group_data):
            duration = float(model['total_time'])
            t.append(duration)
        all_group_data.append(t)
    print(all_group_data)

    x = np.arange(len(all_group_data))
    reshape_datas = [list(x) for x in zip(*all_group_data)]
    print(reshape_datas)

    for idx, data in enumerate(reshape_datas):
        print(data)
        label = MODEL_NAME[idx]
        plt.bar(x+0.25*idx, data, 
                color=color_options[idx % len(color_options)], 
                hatch=hatch_options[idx % len(hatch_options)],
                width=0.25, label=label)

    plt.title('')
    plt.ylabel('Training Duration (seconds)')  
    plt.xlabel('Synchornization Frequency')  
    
    plt.xticks(x + 0.125, ('F=1', 'F=5'))
    plt.legend(loc='upper right', frameon=False)

def plot_volume_group(all_data, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(4*DEFAULT_FIGURE_SIZE, 3*DEFAULT_FIGURE_SIZE)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    all_group_data = []
    for group_data in all_data:
        group_volumes = []
        for idx, model in enumerate(group_data):
            volume = model['transmission_volume']
            group_volumes.append(volume)
        print(group_volumes)
        group_volumes =  [float(x) / 8/1024/1024/1024 for x in group_volumes]         
        all_group_data.append(group_volumes)
    print(all_group_data)

    # exit(0)
    x = np.arange(len(all_group_data))
    reshape_all_accs = [list(x) for x in zip(*all_group_data)]
    print(reshape_all_accs)

    for idx, data in enumerate(reshape_all_accs):
        # print(data)
        label = MODEL_NAME[idx]
        plt.bar(x+0.25*idx, data, 
                color=color_options[idx % len(color_options)], 
                hatch=hatch_options[idx % len(hatch_options)],
                width=0.25, label=label)

    plt.title('')
    plt.ylabel('Transmission Volume (GB)')  
    plt.xlabel('Synchornization Frequency')  
    
    plt.xticks(x + 0.125, ('F=1', 'F=5'))
    plt.legend(loc='upper right', frameon=False)


def plot_best_acc_group(all_data, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(4*DEFAULT_FIGURE_SIZE, 3*DEFAULT_FIGURE_SIZE)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    all_group_data = []
    for group_data in all_data:
        best_accs = [max(model['acc']) for model in group_data]
        all_group_data.append(best_accs)
    print(all_group_data)

    x = np.arange(len(all_group_data))
    reshape_datas = [list(x) for x in zip(*all_group_data)]
    print(reshape_datas)

    for idx, data in enumerate(reshape_datas):
        print(data)
        label = MODEL_NAME[idx]
        plt.bar(x+0.25*idx, data, 
                color=color_options[idx % len(color_options)], 
                hatch=hatch_options[idx % len(hatch_options)],
                width=0.25, label=label)

    plt.title('')
    plt.ylabel('Accuracy')  
    plt.xlabel('Synchornization Frequency')  
    
    plt.xticks(x + 0.125, ('F=1', 'F=5'))
    plt.legend(loc='lower right', frameon=True)

if __name__ == '__main__':
    cmd_args = opt_parser()
    if not cmd_args.load_file:
        print('PATH ERROR')
        sys.exit(0)


    merged_data = get_merged_data(cmd_args.load_file)    
    
    model_type = 'model'
    if 'resnet' in cmd_args.load_file:
        model_type = 'resnet'
    elif 'mobilenet' in cmd_args.load_file:
        model_type = 'mobilenet'

    # Create output folder
    script_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = './post_oral_experiments/02_diff_freq/result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    plot_best_acc_group(all_data=merged_data, figure_idx=1)
    utils.group_plotter.save_figure(os.path.join(output_dir, f'{model_type}_{script_time}_accuracy.png'))

    plot_duration_group(all_data=merged_data, figure_idx=2)
    utils.group_plotter.save_figure(os.path.join(output_dir, f'{model_type}_{script_time}_duration.png'))
    
    plot_volume_group(all_data=merged_data, figure_idx=3)
    utils.group_plotter.save_figure(os.path.join(output_dir, f'{model_type}_{script_time}_trans.png'))

    exit(0)