import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

import numpy as np

from constants import *

def plot_data(data, label):
    plt.plot(data, label=label, marker="o", linestyle="-")

def setup_plot(figure_title, y_axis_label, figure_idx):
    ax1 = plt.figure(figure_idx).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis
    plt.title(figure_title)
    plt.ylabel(y_axis_label)  # y label
    plt.xlabel("Iteration Rounds")  # x label

def legend():
    plt.legend(loc='lower right')

def save_figure(base_dir, filename):
    image_path = os.path.join(base_dir, filename)
    print(image_path)
    plt.savefig(image_path)

def show():
    # plt.legend(loc='lower right')
    # plt.legend(loc='upper left')
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def multiplot(all_data, y_label, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    # plt.figure(num=figure_idx, figsize=(10, 8))
    plt.title(title)
    plt.ylabel(y_label)   # y label
    plt.xlabel("Iteration Rounds")  # x label
    # font = {'size'   : 12}
    # plt.rc('font', **font)
    
    for data in all_data:
        # print(data)
        plt.plot(data.get('acc'),
                 label=data.get('name'),
                 marker="o",
                 linestyle="-")
    return 


def plot_speedup_ratio(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    
    all_ratios = []
    for data in all_data:
        if data.get('speedup_ratio'):
            round_tranmission_ratio = round(float(data['speedup_ratio']), 4)
            # print(round_tranmission_ratio)
            all_ratios.append(round_tranmission_ratio)
        else:
            all_ratios.append(0.0)
    print(all_ratios)

    x = np.arange(1)
    reshape_all_accs = []
    for d in all_ratios:
        reshape_all_accs.append([d])
    # print(reshape_all_accs)

    color_options = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    model_name = [ t['name'] for t in all_data]
    # print(model_name)
    for idx, model_accs in enumerate(reshape_all_accs):
        # print(model_accs)
        plt.bar(x+0.1*idx, model_accs, color=color_options[idx], width=0.1, label=model_name[idx])


    plt.title(title)
    plt.ylabel('Speedup')  
    plt.xlabel('')  
    plt.xticks([])

    plt.legend(loc='upper left')


def plot_best_acc(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    
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
    # print(reshape_all_accs)

    color_options = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    model_name = [ t['name'] for t in all_data]
    # print(model_name)
    for idx, model_accs in enumerate(reshape_all_accs):
        # print(model_accs)
        plt.bar(x+0.1*idx, model_accs, color=color_options[idx], width=0.1, label=model_name[idx])

    plt.title(title)
    plt.ylabel('Accuracy')  
    plt.xlabel('')
    plt.xticks([])  
    plt.legend(loc='lower right')


def plot_transmission_ratio(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    
    all_volume = []
    for model in all_data:
        if model.get('transmission_volume'):
            volume = model['transmission_volume']
            all_volume.append(volume)
            if 'Baseline' in model['name']:
                baseline_volume = volume
                # print(baseline_volume)
        else:
            all_volume.append(0.0)
    print(all_volume)

    all_volume =  [float(x) / int(baseline_volume) for x in all_volume] 
    print(all_volume)

    x = np.arange(1)
    reshape_data = []
    for d in all_volume:
        reshape_data.append([d])
    # print(reshape_data)

    color_options = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    model_name = [ t['name'] for t in all_data]
    # print(model_name)
    for idx, model_volume in enumerate(reshape_data):
        # print(model_volume)
        plt.bar(x+0.1*idx, model_volume, color=color_options[idx], width=0.1, label=model_name[idx])

    plt.title(title)
    plt.ylabel('Transmission Volume Overhead')  
    plt.xlabel('')
    plt.xticks([])  
    plt.legend(loc='lower right')


def plot_trans_to_acc(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    plt.title(title)
    plt.ylabel("Accuracy")   # y label
    plt.xlabel("Transmission volume (MB)")  # x label
    # font = {'size'   : 12}
    # plt.rc('font', **font)

    for data in all_data:
        # print(data['transmission_volume_history'])
        transmission_volumes = eval(data['transmission_volume_history'])

        data_transmission_in_mb = [float(x)/8/1024/1024 for x in transmission_volumes]
        # print(data_transmission_in_mb)
        
        plt.plot(data_transmission_in_mb, data['acc'][WARM_UP_ROUNDS:], 
                 label=data.get('name'),
                 marker="o",
                 linestyle="-")
    
    plt.legend(loc='lower right')
 
def scatter_trans_to_bestacc(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    plt.title(title)
    plt.ylabel("Accuracy")   # y label
    plt.xlabel("Transmission volume (GB)")  # x label
    # font = {'size'   : 12}
    # plt.rc('font', **font)

    for data in all_data:
        # print(data['transmission_volume_history'])
        # transmission_volumes = eval(data['transmission_volume_history'])
        # data_transmission_in_mb = [float(x)/8/1024/1024 for x in transmission_volumes]
        # print(data_transmission_in_mb)
        
        best_acc = round(max(data['acc']), 4)
        trans_volume = float(data['transmission_volume']) / 8/ 1024/ 1024 / 1024
        print(f'{best_acc} {trans_volume}')

        plt.scatter(trans_volume, best_acc, 
                    label=data.get('name'),
                    marker="o",
                    linestyle="-")

    plt.legend(loc='lower right')