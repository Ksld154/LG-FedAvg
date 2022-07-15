from cProfile import label
from matplotlib import markers
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

import numpy as np

from constants import *

DEFAULT_FIGURE_SIZE = 1.5
color_options  = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
linestyle_options = ['-', '--', '-.', ':']
marker_options = ['o', '*', '.', ',', 'x', 'P', 'D', 'H']
hatch_options = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
MODEL_NAME = ['threshold=0.1', 'threshold=0.05', 'threshold=0.03', 'threshold=0.01', 'threshold=0.005', 'threshold=0.003', 'threshold=0.001', "No Freeze", "With Opportunistic Switch"]


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
        if 'Static: all' in data['name']:
            continue
        
        plt.plot(data.get('acc'),
                 label=data['name'],
                 marker=marker_options[idx % len(marker_options)],
                 linestyle=linestyle_options[idx % len(linestyle_options)])
    legend() 
    # plt.ylim(bottom=0.0)

def plot_epoch_to_trans(all_data, y_label, title, figure_idx):
    set_figure_size(figure_idx=figure_idx)

    plt.title(title)
    plt.ylabel("Transmission Volume Overhead (MB)")   # y label
    plt.xlabel("Iteration Rounds")  # x label
    
    for idx, data in enumerate(all_data):
        if 'Static: all' in data['name']:
            continue
        transmission_volumes = eval(data['transmission_volume_history'])
        # print(f'{data["transmission_volume_history"]} {data["name"]}' )
        data_transmission_in_mb = [float(x)/8/1024/1024 for x in transmission_volumes]
        transmisiion_volume_per_round = [data_transmission_in_mb[i]-data_transmission_in_mb[i-1] for i in range(1, len(data_transmission_in_mb))]
        
        plt.plot(transmisiion_volume_per_round,
                 label=data['name'],
                 marker=marker_options[idx % len(marker_options)],
                 linestyle=linestyle_options[idx % len(linestyle_options)])
    plt.legend(loc='lower left', frameon=False)
    plt.ylim(bottom=0)


def plot_speedup_ratio(all_data, title, figure_idx):
    set_figure_size(figure_idx=figure_idx)

    all_ratios = []
    model_name = []
    for idx, data in enumerate(all_data):
        model_name.append(data['name'])
        if data.get('speedup_ratio') and 'Static: all' not in data['name']:
            round_tranmission_ratio = round(float(data['speedup_ratio']), 4)
            all_ratios.append(round_tranmission_ratio)
    print(all_ratios)

    x = np.arange(1)
    reshape_all_accs = []
    for d in all_ratios:
        reshape_all_accs.append([d])

    for idx, model_accs in enumerate(reshape_all_accs):
        plt.bar(x+0.1*idx, model_accs, 
                color=color_options[idx % len(color_options)], 
                hatch=hatch_options[idx % len(hatch_options)],
                width=0.1, label=model_name[idx])

    plt.title(title)
    plt.ylabel('Speedup')  
    plt.xlabel('')
    plt.ylim([0.0, 2.5])  
    plt.xticks([])
    plt.legend(loc='upper left', frameon=False)


def plot_transmission_ratio(all_data, title, figure_idx):
    set_figure_size(figure_idx=figure_idx)

    all_volume = []
    model_name = []
    for idx, model in enumerate(all_data):
        model_name.append(model['name'])
        if model.get('transmission_volume') and 'Static: all' not in model['name']:
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
        label = model_name[idx]
        plt.bar(x+0.1*idx, model_volume, color=color_options[idx % len(color_options)], hatch=hatch_options[idx % len(hatch_options)], width=0.1, label=label)

    plt.title(title)
    plt.ylabel('Transmission Volume Overhead Ratio')  
    plt.xlabel('')
    plt.xticks([])
    plt.ylim([0.0, 2.0])  
    plt.legend(loc='upper right', frameon=False)


def plot_trans_to_acc(all_data, title, figure_idx):
    set_figure_size(figure_idx=figure_idx)

    plt.title(title)
    plt.ylabel("Accuracy")   # y label
    plt.xlabel("Transmission volume (GB)")  # x label

    for idx, data in enumerate(all_data):
        transmission_volumes = eval(data['transmission_volume_history'])
        data_transmission_in_gb = [float(x)/8/1024/1024/1024 for x in transmission_volumes]
        plt.plot(data_transmission_in_gb, data['acc'][WARM_UP_ROUNDS:], 
                #  label=data.get('name'),
                 label=data['name'],
                 marker=marker_options[idx % len(marker_options)],
                 linestyle=linestyle_options[idx % len(linestyle_options)])
    plt.legend(loc='upper left', frameon=False)
 

def plot_best_acc(all_data, title, figure_idx):
    set_figure_size(figure_idx=figure_idx)
    
    all_accs = []
    for model in all_data:
        if model.get('acc'):
            round_best_acc = round(max(model['acc']), 4)
            all_accs.append(round_best_acc)
        else:
            all_accs.append(0.0)
    print(all_accs)

    x = np.arange(1)
    reshape_all_accs = []
    for d in all_accs:
        reshape_all_accs.append([d])

    model_name = [ t['name'] for t in all_data]
    for idx, model_accs in enumerate(reshape_all_accs):
        plt.bar(x+0.1*idx, model_accs, color=color_options[idx % len(color_options)], hatch=hatch_options[idx % len(hatch_options)], width=0.1, label=model_name[idx])

    plt.title(title)
    plt.ylabel('Accuracy')  
    plt.xlabel('')
    plt.xticks([])
    plt.ylim(bottom=0.0)  
    leg = plt.legend(loc='upper right', frameon=False, ncol=2)
    leg.set_draggable(state=True)    


def plot_trans_to_target_acc(all_data, title, figure_idx, model_type):
    set_figure_size(figure_idx=figure_idx)
    if model_type == 'ResNet-18':
        target_accs = [0.7, 0.75, 0.8, 0.85, 0.9]
    elif model_type == 'MobileNet':
        target_accs = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    else:
        target_accs = [0.65, 0.7, 0.75, 0.8]


    print(target_accs)
    all_group_data = []
    model_name = []
    for idx, data in enumerate(all_data):
        target_acc_idx = 0
        if 'Static: all' in data['name']:
            continue  
        model_name.append(data['name'])
        
        transmission_volumes = eval(data['transmission_volume_history'])
        data_transmission_in_mb = [float(x)/8/1024/1024/1024 for x in transmission_volumes]
        print(len(data_transmission_in_mb))
        # print(data_transmission_in_mb)
        if 'No Freeze' in data["name"]:
            data['acc'] = data['acc'][WARM_UP_ROUNDS:]
        
        accs = data['acc']
        print(len(accs))

        target_trans_volume = []
        for idx, acc in enumerate(accs):
            if target_acc_idx < len(target_accs) and acc >= target_accs[target_acc_idx]:
                target_acc_idx += 1
                print(f'{acc}, {idx}')
                target_trans_volume.append(data_transmission_in_mb[idx])

        print(target_trans_volume)

        for i in range(len(target_accs) - len(target_trans_volume)):
            target_trans_volume.append(0)
        all_group_data.append(target_trans_volume)    
    print(all_group_data)
    
    x = np.arange(len(target_accs))
    # reshaped_data = [list(x) for x in zip(*all_group_data)]
    # print(reshaped_data)

    for idx, model_volume in enumerate(all_group_data):
        label = model_name[idx]

        plt.bar(x+0.1*idx, model_volume, color=color_options[idx % len(color_options)], hatch=hatch_options[idx % len(hatch_options)], width=0.1, label=label)

    plt.title(title)
    plt.ylabel('Transmission Volume Overhead (GB)')  
    plt.xlabel('Target Accuracy')

    xtick_label = [f'{acc}' for acc in target_accs]
    plt.xticks(x+0.2, xtick_label)  
    plt.legend(loc='upper left', frameon=False)


def plot_accuracy_to_transmission_volume_ratio(all_data, model_name, figure_idx):
    set_figure_size(figure_idx=figure_idx)

    resnet_data = all_data[0]
    mobilenet_data = all_data[1]

    x = np.arange(len(all_data))
    grouped_data = []
    for _, (r_data, m_data) in enumerate(zip(resnet_data, mobilenet_data)):
        grouped_data.append([r_data, m_data])
    print(grouped_data)

    # x = np.arange(len(grouped_data))
    # reshape_data = [list(x) for x in zip(*grouped_data)]
    # print(reshape_data)

    for idx, ratio in enumerate(grouped_data):
        print(ratio)
        plt.bar(x+0.1*idx, ratio, 
            color=color_options[idx % len(color_options)], 
            hatch=hatch_options[idx % len(hatch_options)], 
            width=0.1, label=model_name[idx])


    plt.ylabel('Best Accuracy / Transmission Volume (GB)')  
    plt.xlabel('')
    plt.xticks(x + 0.3, ('ResNet-18', 'MobileNet'))
    leg = plt.legend(loc='upper left', frameon=False, ncol=3)
    leg.set_draggable(state=True)    


def plot_total_transmission(all_data, title, figure_idx):
    set_figure_size(figure_idx=figure_idx)

    all_volume = [ float(model['transmission_volume'])/8/1024/1024/1024 for model in all_data]

    x = np.arange(1)
    reshape_data = []
    for d in all_volume:
        reshape_data.append([d])

    model_name = [ t['name'] for t in all_data]
    for idx, model_volume in enumerate(reshape_data):
        label = model_name[idx]
        plt.bar(x+0.1*idx, model_volume, color=color_options[idx % len(color_options)], hatch=hatch_options[idx % len(hatch_options)], width=0.1, label=label)

    plt.title(title)
    plt.ylabel('Total Transmission Volume (GB)')  
    plt.xlabel('')
    plt.xticks([])
    # plt.ylim([0.0, 30.0])  
    
    leg = plt.legend(loc='upper right', frameon=False, ncol=2)
    leg.set_draggable(state=True)    
    

def plot_epoch_to_target_acc(all_data, title, figure_idx, model_type):
    set_figure_size(figure_idx=figure_idx)
    if model_type == 'ResNet-18':
        target_accs = [0.7, 0.75, 0.8, 0.85, 0.9]
    elif model_type == 'MobileNet':
        target_accs = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    else: # merged
        target_accs = [0.65, 0.7, 0.75, 0.8]


    print(target_accs)
    all_group_data = []
    model_name = []
    for idx, data in enumerate(all_data):
        target_acc_idx = 0
        if 'Static: all' in data['name']:
            continue  
        model_name.append(data['name'])

        
        if 'No Freeze' in data["name"]:
            data['acc'] = data['acc'][WARM_UP_ROUNDS:]
        
        accs = data['acc']
        print(len(accs))

        target_epoch = []
        for idx, acc in enumerate(accs):
            if target_acc_idx < len(target_accs) and acc >= target_accs[target_acc_idx]:
                target_acc_idx += 1
                print(f'{acc}, {idx}')
                target_epoch.append(idx)
        print(target_epoch)

        for i in range(len(target_accs) - len(target_epoch)):
            target_epoch.append(0)
        all_group_data.append(target_epoch)    
    print(all_group_data)
    
    x = np.arange(len(target_accs))


    for idx, model_volume in enumerate(all_group_data):
        label = model_name[idx]
        plt.bar(x+0.1*idx, model_volume, color=color_options[idx % len(color_options)], hatch=hatch_options[idx % len(hatch_options)], width=0.1, label=label)

    plt.title(title)
    plt.ylabel('Iteration Rounds')  
    plt.xlabel('Target Accuracy')

    xtick_label = [f'{acc}' for acc in target_accs]
    plt.xticks(x+0.2, xtick_label)  
    plt.legend(loc='upper left', frameon=False)