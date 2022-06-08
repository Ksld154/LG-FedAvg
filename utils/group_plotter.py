from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
import os

def multiplot(all_data, y_label, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    # plt.figure(num=figure_idx, figsize=(12, 8))
    plt.title(title)
    plt.ylabel(y_label)   # y label
    plt.xlabel("Epochs")  # x label
    font = {'size'   : 12}
    plt.rc('font', **font)

    for data in all_data:
        # print(data)
        plt.plot(data.get('acc'),
                 label=data.get('name'),
                 marker="o",
                 linestyle="-")
    plt.legend(loc='lower right')

def show():
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def save_figure(filepath:str):
    plt.savefig(filepath)
    # print(filepath)

def plot_best_acc(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    # model_names = [ t['name'] for t in all_data]
    # print(model_names)
    
    all_group_best_accs = []
    for group_data in all_data:
        best_accs = [ max(model['acc'].strip('][').split(', ')) for model in group_data]
        round_best_accs = [round(float(x), 4) for x in best_accs]
        all_group_best_accs.append(round_best_accs)
        print(round_best_accs)
    
    print(all_group_best_accs)
    x = np.arange(len(all_group_best_accs))
    print(x)

    reshape_all_accs = [list(x) for x in zip(*all_group_best_accs)]
    print(reshape_all_accs)

    color_options = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    model_name = [ t['name'] for t in all_data[0]]
    print(model_name)
    for idx, model_accs in enumerate(reshape_all_accs):
        print(model_accs)
        plt.bar(x+0.1*idx, model_accs, color=color_options[idx], width=0.1, label=model_name[idx])
        # plt.xticks(x, model_names)

    plt.ylim([0, 1])

    plt.title(title)
    plt.ylabel('Accuracy')  
    plt.xlabel('Initial Freezing Timeslot')  
    # font = {'size': 12}
    # plt.rc('font', **font)
    
    plt.xticks(x + 0.2, ('I=0.1', 'I=0.25'))
    plt.legend(loc='lower right')


def plot_transmission_ratio(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    
    all_group_data = []
    for group_data in all_data:
        # print(group_data)
        group_volumes = []
        for model in group_data:
            volume = model['transmission_volume']
            if 'Baseline' in model['name']:
                baseline_volume = volume
            group_volumes.append(volume)
        # group_volumes =  [float(x) / int(baseline_volume) for x in group_volumes]
        group_volumes =  [float(x) / 8/1024/1024/1024 for x in group_volumes]         

        all_group_data.append(group_volumes)
    print(all_group_data)

    # exit(0)
    x = np.arange(len(all_group_data))
    reshape_all_accs = [list(x) for x in zip(*all_group_data)]
    print(reshape_all_accs)

    color_options = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    model_name = [ t['name'] for t in all_data[0]]
    print(model_name)
    for idx, model_accs in enumerate(reshape_all_accs):
        print(model_accs)
        plt.bar(x+0.1*idx, model_accs, color=color_options[idx], width=0.1, label=model_name[idx])
        # plt.xticks(x, model_names)

    # plt.ylim([0.7, 0.9])

    plt.title(title)
    plt.ylabel('Transmission Volume (GB)')  
    plt.xlabel('Worker Selection Ratio')  
    # font = {'size': 12}
    # plt.rc('font', **font)
    
    plt.xticks(x + 0.2, ('C=0.5', 'C=1.0'))
    plt.legend(loc='lower right')

def calc_transmission_speedup(all_data):
    baseline_time = 0
    result_all_data = []
    for idx, data in enumerate(all_data):
        if data.get('total_training_time'):
            pt = datetime.datetime.strptime(data['total_training_time'],'%H:%M:%S.%f')
            tt = pt - datetime.datetime(1900, 1, 1)
            total_seconds = tt.total_seconds()
            print(total_seconds)
            data['total_training_time'] = total_seconds
            
            if 'Baseline' in data['name']:
                baseline_time = total_seconds
                print(baseline_time)
            
            
    for data in all_data:
        data['speedup_ratio'] = baseline_time / data['total_training_time']
        result_all_data.append(data)
    print(result_all_data)
    return result_all_data


def plot_speedup_ratio(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    all_group_data = []
    for group_data in all_data:
        speedup_ratio = [ round(float(model['speedup_ratio']), 4) for model in group_data]
        print(speedup_ratio)
        # round_tranmission_ratio = [round(float(x), 4) for x in tranmission_ratio]
        all_group_data.append(speedup_ratio)
    print(all_group_data)


    x = np.arange(len(all_group_data))
    reshape_all_group_data = [list(x) for x in zip(*all_group_data)]
    print(reshape_all_group_data)

    color_options = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    model_name = [ t['name'] for t in all_data[0]]
    print(model_name)
    for idx, model_speedup in enumerate(reshape_all_group_data):
        print(model_speedup)
        plt.bar(x+0.1*idx, model_speedup, color=color_options[idx], width=0.1, label=model_name[idx])
        # plt.xticks(x, model_names)

    plt.title(title)
    plt.ylabel('Speedup')  
    plt.xlabel('Initial Freezing Timeslot')  
    plt.xticks([])
    # font = {'size': 12}
    # plt.rc('font', **font)
    
    plt.xticks(x + 0.2, ('I=0.1', 'I=0.25'))
    plt.legend(loc='lower right')