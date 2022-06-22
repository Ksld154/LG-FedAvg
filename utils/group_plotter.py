import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
import os

DEFAULT_FIGURE_SIZE = 1.5
color_options  = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
linestyle_options = ['-', '--', '-.', ':']
marker_options = ['o', '*', '.', ',', 'x', 'P', 'D', 'H']
hatch_options = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
MODEL_NAME = ['Baseline', 'Static: 1/4', 'Static: 1/2', 'Static: 3/4', 'Static: all', 'Gradually Freezing']



def save_figure(filepath:str):
    plt.savefig(filepath)


def plot_transmission_volume(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(4*DEFAULT_FIGURE_SIZE, 3*DEFAULT_FIGURE_SIZE)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    all_group_data = []
    for group_data in all_data:
        group_volumes = []
        for idx, model in enumerate(group_data):
            if 'Static: all' not in MODEL_NAME[idx]:
                volume = model['transmission_volume']
                group_volumes.append(volume)

        group_volumes =  [float(x) / 8/1024/1024/1024 for x in group_volumes]         
        all_group_data.append(group_volumes)
    print(all_group_data)

    # exit(0)
    x = np.arange(len(all_group_data))
    reshape_all_accs = [list(x) for x in zip(*all_group_data)]
    print(reshape_all_accs)

    for idx, data in enumerate(reshape_all_accs):
        print(data)
        label = MODEL_NAME[idx+1] if 'Static: all' in MODEL_NAME[idx] else MODEL_NAME[idx]
        plt.bar(x+0.1*idx, data, 
                color=color_options[idx % len(color_options)], 
                hatch=hatch_options[idx % len(hatch_options)],
                width=0.1, label=label)

    plt.title(title)
    plt.ylabel('Transmission Volume (GB)')  
    plt.xlabel('Worker Selection Ratio')  
    
    plt.xticks(x + 0.2, ('S=0.5', 'S=1.0'))
    plt.legend(loc='upper left', frameon=False)





def calc_transmission_speedup(all_data):
    baseline_time = 0
    result_all_data = []
    for idx, data in enumerate(all_data):
        if 'Baseline' in data['name']:
            baseline_time = float(data['total_time'])
            print(baseline_time)
            
            
    for data in all_data:
        data['speedup_ratio'] = baseline_time / float(data['total_time'])
        result_all_data.append(data)
    print(result_all_data)
    return result_all_data






def plot_speedup_ratio(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(4*DEFAULT_FIGURE_SIZE, 3*DEFAULT_FIGURE_SIZE)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    all_group_data = []
    for group_data in all_data:
        group_volumes = []
        for idx, model in enumerate(group_data):
            if 'Static: all' not in MODEL_NAME[idx]:
                volume = float(model['speedup_ratio'])
                group_volumes.append(volume)
        all_group_data.append(group_volumes)
    print(all_group_data)

    x = np.arange(len(all_group_data))
    reshape_datas = [list(x) for x in zip(*all_group_data)]
    print(reshape_datas)

    for idx, data in enumerate(reshape_datas):
        print(data)
        label = MODEL_NAME[idx+1] if 'Static: all' in MODEL_NAME[idx] else MODEL_NAME[idx]
        plt.bar(x+0.1*idx, data, 
                color=color_options[idx % len(color_options)], 
                hatch=hatch_options[idx % len(hatch_options)],
                width=0.1, label=label)

    plt.title(title)
    plt.ylabel('Training Speed-up Ratio')  
    plt.xlabel('Worker Selection Ratio')  
    
    plt.xticks(x + 0.2, ('S=0.5', 'S=1.0'))
    plt.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.55, 1))

def plot_speedup_duration(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(4*DEFAULT_FIGURE_SIZE, 3*DEFAULT_FIGURE_SIZE)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    all_group_data = []
    for group_data in all_data:
        group_volumes = []
        for idx, model in enumerate(group_data):
            if 'Static: all' not in MODEL_NAME[idx]:
                volume = float(model['total_time'])
                group_volumes.append(volume)
        all_group_data.append(group_volumes)
    print(all_group_data)

    x = np.arange(len(all_group_data))
    reshape_datas = [list(x) for x in zip(*all_group_data)]
    print(reshape_datas)

    for idx, data in enumerate(reshape_datas):
        print(data)
        label = MODEL_NAME[idx+1] if 'Static: all' in MODEL_NAME[idx] else MODEL_NAME[idx]
        plt.bar(x+0.1*idx, data, 
                color=color_options[idx % len(color_options)], 
                hatch=hatch_options[idx % len(hatch_options)],
                width=0.1, label=label)

    plt.title(title)
    plt.ylabel('Training Duration (seconds)')  
    plt.xlabel('Worker Selection Ratio')  
    
    plt.xticks(x + 0.2, ('S=0.5', 'S=1.0'))
    plt.legend(loc='upper left', frameon=False)