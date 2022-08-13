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
MODEL_NAME = ['Baseline', 'Static: 1/4', 'Static: 1/2', 'Static: 3/4', 'Static: all', 'Gradually Freezing']

MODEL_NAME2 = ['Baseline', 'Layer Freezing w/ PCME']


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

def multiplot(all_data, y_label, title, figure_idx):
    set_figure_size(figure_idx=figure_idx)

    plt.title(title)
    plt.ylabel(y_label)   # y label
    plt.xlabel("Iteration Rounds")  # x label
    
    for idx, data in enumerate(all_data):
        if 'Static: all' in MODEL_NAME[idx]:
            continue
        
        plt.plot(data.get('acc'),
                 label=MODEL_NAME[idx],
                 marker=marker_options[idx % len(marker_options)],
                 linestyle=linestyle_options[idx % len(linestyle_options)])
    # plt.ylim(bottom=0.0)
    leg = plt.legend(loc='lower right', frameon=False)
    leg.set_draggable(state=True)    

def plot_epoch_to_trans(all_data, prefreeze_volume, title, figure_idx):
    set_figure_size(figure_idx=figure_idx)

    plt.title(title)
    plt.ylabel("Transmission Volume Overhead (MB)")   # y label
    plt.xlabel("Iteration Rounds")  # x label
    
    for idx, data in enumerate(all_data):
        if 'Static: all' in MODEL_NAME[idx]:
            continue
        transmission_volumes = eval(data['transmission_volume_history'])
        data_transmission_in_mb = [float(x)/8/1024/1024 for x in transmission_volumes]
        transmisiion_volume_per_round = [prefreeze_volume]*WARM_UP_ROUNDS + [data_transmission_in_mb[i]-data_transmission_in_mb[i-1] for i in range(1, len(data_transmission_in_mb))]
        
        plt.plot(transmisiion_volume_per_round,
                 label=MODEL_NAME[idx],
                 marker=marker_options[idx % len(marker_options)],
                 linestyle=linestyle_options[idx % len(linestyle_options)])
    leg = plt.legend(loc='lower left', frameon=False)
    leg.set_draggable(state=True)    


def plot_speedup_ratio(all_data, title, figure_idx):
    set_figure_size(figure_idx=figure_idx)

    all_ratios = []
    for idx, data in enumerate(all_data):
        if data.get('speedup_ratio') and 'Static: all' not in MODEL_NAME[idx]:
            round_tranmission_ratio = round(float(data['speedup_ratio']), 4)
            all_ratios.append(round_tranmission_ratio)
    print(all_ratios)

    x = np.arange(1)
    reshape_all_accs = []
    for d in all_ratios:
        reshape_all_accs.append([d])

    for idx, model_accs in enumerate(reshape_all_accs):
        label = MODEL_NAME[idx+1] if 'Static: all' in MODEL_NAME[idx] else MODEL_NAME[idx]
            
        plt.bar(x+0.1*idx, model_accs, 
                color=color_options[idx % len(color_options)], 
                hatch=hatch_options[idx % len(hatch_options)],
                width=0.1, label=label)

    plt.title(title)
    plt.ylabel('Speedup')  
    plt.xlabel('')  
    plt.xticks([])
    plt.legend(loc='upper left', frameon=False)



def plot_transmission_ratio(all_data, title, figure_idx):
    set_figure_size(figure_idx=figure_idx)

    all_volume = []
    for idx, model in enumerate(all_data):
        if model.get('transmission_volume') and 'Static: all' not in MODEL_NAME[idx]:
            volume = model['transmission_volume']
            all_volume.append(volume)
            if 'Baseline' in model['name']:
                baseline_volume = volume
    print(all_volume)

    all_volume =  [float(x) / int(baseline_volume) for x in all_volume] 
    print(all_volume)

    x = np.arange(1)
    reshape_data = []
    for d in all_volume:
        reshape_data.append([d])

    model_name = [ t['name'] for t in all_data]
    for idx, model_volume in enumerate(reshape_data):
        label = MODEL_NAME[idx+1] if 'Static: all' in MODEL_NAME[idx] else MODEL_NAME[idx]

        plt.bar(x+0.1*idx, model_volume, color=color_options[idx % len(color_options)], hatch=hatch_options[idx % len(hatch_options)], width=0.1, label=label)

    plt.title(title)
    plt.ylabel('Transmission Volume Overhead')  
    plt.xlabel('')
    plt.xticks([])  
    plt.legend(loc='upper right', frameon=False)


def plot_trans_to_acc(all_data, title, figure_idx):
    set_figure_size(figure_idx=figure_idx)

    plt.title(title)
    plt.ylabel("Accuracy")   # y label
    plt.xlabel("Transmission volume (GB)")  # x label

    for idx, data in enumerate(all_data):
        transmission_volumes = eval(data['transmission_volume_history'])
        data_transmission_in_mb = [float(x)/8/1024/1024/1024 for x in transmission_volumes]
        # print(data_transmission_in_mb)
        
        plt.plot(data_transmission_in_mb, data['acc'][WARM_UP_ROUNDS:], 
                #  label=data.get('name'),
                 label=MODEL_NAME[idx],
                 marker=marker_options[idx % len(marker_options)],
                 linestyle=linestyle_options[idx % len(linestyle_options)])
    plt.legend(loc='upper left', frameon=False)
 

def plot_trans_to_target_acc(all_data, title, figure_idx, model_type):
    set_figure_size(figure_idx=figure_idx)
    if model_type == 'ResNet-18':
        target_accs = [0.7, 0.75, 0.8, 0.85, 0.9]
    elif model_type == 'MobileNet':
        target_accs = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        # target_accs = [0.7, 0.75, 0.8, 0.85, 0.9]


    print(target_accs)
    all_group_data = []
    for idx, data in enumerate(all_data):
        target_acc_idx = 0
        if 'Static: all' in MODEL_NAME[idx]:
            continue  
        
        transmission_volumes = eval(data['transmission_volume_history'])
        data_transmission_in_gb = [float(x)/8/1024/1024/1024 for x in transmission_volumes]
        # print(data_transmission_in_mb)
        accs = data['acc']

        target_trans_volume = []
        for idx, acc in enumerate(accs):
            if idx < WARM_UP_ROUNDS:
                continue
            
            if target_acc_idx < len(target_accs) and acc >= target_accs[target_acc_idx]:
                target_acc_idx += 1
                print(f'{acc}, {idx}')
                target_trans_volume.append(data_transmission_in_gb[idx-WARM_UP_ROUNDS])

        print(target_trans_volume)

        for i in range(len(target_accs) - len(target_trans_volume)):
            target_trans_volume.append(0)
        all_group_data.append(target_trans_volume)    
    print(all_group_data)
    
    x = np.arange(len(target_accs))
    # reshaped_data = [list(x) for x in zip(*all_group_data)]
    # print(reshaped_data)

    for idx, model_volume in enumerate(all_group_data):
        label = MODEL_NAME[idx+1] if 'Static: all' in MODEL_NAME[idx] else MODEL_NAME[idx]

        plt.bar(x+0.1*idx, model_volume, color=color_options[idx % len(color_options)], hatch=hatch_options[idx % len(hatch_options)], width=0.1, label=label)

    plt.title(title)
    plt.ylabel('Transmission Volume Overhead (GB)')  
    plt.xlabel('Target Accuracy')

    xtick_label = [f'{acc}' for acc in target_accs]
    plt.xticks(x+0.2, xtick_label)  
    leg = plt.legend(loc='upper left', frameon=False)
    leg.set_draggable(state=True)    

def plot_duration(all_data, title, figure_idx):
    set_figure_size(figure_idx=figure_idx)

    all_duration = []
    for idx, data in enumerate(all_data):
        if 'Static: all' not in MODEL_NAME[idx]:
            d = float(data['total_time'])
            all_duration.append(d)
    print(all_duration)

    x = np.arange(1)
    reshape_all_accs = []
    for d in all_duration:
        reshape_all_accs.append([d])

    for idx, model_accs in enumerate(reshape_all_accs):
        label = MODEL_NAME[idx+1] if 'Static: all' in MODEL_NAME[idx] else MODEL_NAME[idx]
        
           
        plt.bar(x+0.1*idx, model_accs, 
                color=color_options[idx % len(color_options)], 
                hatch=hatch_options[idx % len(hatch_options)],
                width=0.1, label=label)

    plt.title(title)
    plt.ylabel('Duration (seconds)')  
    plt.xlabel('')  
    plt.xticks([])
    plt.legend(loc='upper right', frameon=False)

def plot_volume(all_data, title, figure_idx):
    set_figure_size(figure_idx=figure_idx)

    all_duration = []
    for idx, data in enumerate(all_data):
        if 'Static: all' not in MODEL_NAME[idx]:
            d = float(data['transmission_volume'])
            d_mb = float(d) /8/1024/1024
            all_duration.append(d_mb)
    print(all_duration)

    x = np.arange(1)
    reshape_all_accs = []
    for d in all_duration:
        reshape_all_accs.append([d])

    for idx, model_accs in enumerate(reshape_all_accs):
        label = MODEL_NAME[idx+1] if 'Static: all' in MODEL_NAME[idx] else MODEL_NAME[idx]
        
           
        plt.bar(x+0.1*idx, model_accs, 
                color=color_options[idx % len(color_options)], 
                hatch=hatch_options[idx % len(hatch_options)],
                width=0.1, label=label)

    plt.title(title)
    plt.ylabel('Transmission Volume (MB)')  
    plt.xlabel('')  
    plt.xticks([])
    plt.legend(loc='upper right', frameon=False)