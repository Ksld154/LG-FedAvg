import argparse
import sys
import os
import datetime

import utils.csv_exporter
import utils.myplotter
import utils.eps_plotter
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
    parser.add_argument('--load_static_path', type=str, default='', help='Load Static Freeze results from path')
    parser.add_argument('--load_gf_path', type=str, default='', help='Load Gradually Freeze results from path')
    
    return parser.parse_args()


def get_static_freeze_data(filepath, pre_freeze_params):
    data = utils.csv_exporter.import_csv(filepath=filepath)
    return data

def get_gradually_freeze_data_2(filepath, static_freeze_data):
    pretrained_acc = static_freeze_data[0]['acc'][:3]
    # print(pretrained_acc)
    
    data = utils.csv_exporter.import_csv(filepath=filepath)
    # print(data)
    new_data = []
    for row in data:
        gf_acc = [float(d)  for d in row['acc']]
        acc = pretrained_acc + gf_acc
        new_data.append(dict(name="Gradually Freezing: Primary Model", 
                                acc=acc, 
                                total_time=row['total_time'] , 
                                total_trainable_params=row['total_trainable_params'],
                                transmission_time=row['transmission_time'],
                                transmission_volume=row['transmission_volume'],
                                transmission_volume_readable=row['transmission_volume_readable'],
                                transmission_volume_history=row['transmission_volume_history']))
    # print(new_data)
    return new_data

def calc_speedup(all_data):
    baseline_time = 0
    for idx,  data in enumerate(all_data):
        if data.get('total_time'):
            pt = datetime.datetime.strptime(data['total_time'],'%H:%M:%S.%f')
            tt = pt - datetime.datetime(1900, 1, 1)
            total_seconds = tt.total_seconds()
            # print(total_seconds)
            data['total_time'] = total_seconds
            if 'Baseline' in data['name']:
                baseline_time = total_seconds
                print(baseline_time)
        if data.get('transmission_time'):
            pt = datetime.datetime.strptime(data['transmission_time'],'%H:%M:%S.%f') - datetime.datetime(1900, 1, 1)
            total_trans_seconds = pt.total_seconds()
            # print(total_trans_seconds)
            data['transmission_time'] = total_trans_seconds
            if 'Baseline' in data['name']:
                baseline_trans_time = total_trans_seconds
                # print(baseline_trans_time)
        print(f'Total time: {total_seconds} Total transmission time: {total_trans_seconds}')

    result_all_data = []
    for data in all_data:
        data['speedup_ratio'] = baseline_time / data['total_time']
        data['speedup_trans_ratio'] = baseline_trans_time / data['transmission_time']
        print(f'{data["speedup_ratio"]} {data["speedup_trans_ratio"]}')
        
        result_all_data.append(data)
    
    return result_all_data


def plot_epoch_to_accuracy(all_data, output_dir, timestamp, model_type):
    set_figure_size(figure_idx=1)

    plt.title('')
    plt.ylabel("Accuracy")   # y label
    plt.xlabel("Iteration Rounds")  # x label
    
    for idx, data in enumerate(all_data):
        plt.plot(data.get('acc'),
                 label=MODEL_NAME[idx],
                 marker=marker_options[idx % len(marker_options)],
                 linestyle=linestyle_options[idx % len(linestyle_options)])

    leg = plt.legend(loc='lower right', frameon=False)
    leg.set_draggable(state=True)  
    utils.eps_plotter.legend()
    # utils.eps_plotter.save_figure(output_dir, f"{timestamp}_{model_type}_FL_epoch_to_Accuracy.eps")
    utils.eps_plotter.save_figure(output_dir, f"{timestamp}_{model_type}_FL_epoch_to_Accuracy.png")


def plot_duration(all_data, output_dir, model_type):
    set_figure_size(figure_idx=2)

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

    plt.title('')
    plt.ylabel('Duration (seconds)')  
    plt.xlabel('')  
    plt.xticks([])
    plt.legend(loc='upper right', frameon=False)
    utils.eps_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_duration.png') 

def plot_volume(all_data, output_dir, model_type):
    set_figure_size(figure_idx=3)

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

    plt.title('')
    plt.ylabel('Transmission Volume (MB)')  
    plt.xlabel('')  
    plt.xticks([])
    plt.legend(loc='upper right', frameon=False)
    utils.eps_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_trans_volume.png') 


if __name__ == '__main__':
    cmd_args = opt_parser()
    if not cmd_args.load_static_path or not cmd_args.load_gf_path:
        print('PATH ERROR')
        sys.exit(0)
    
    if 'resnet' in cmd_args.load_static_path:
        model_type = 'ResNet-18'
        pre_freeze_params = RESNET_18_PARAMS
    elif 'mobilenet' in cmd_args.load_static_path:
        model_type = 'MobileNet'
        pre_freeze_params = MOBILENET_PARAMS
    else:
        model_type = 'LeNet-5'
        pre_freeze_params = LENET_5_PARAMS


    static_data = get_static_freeze_data(cmd_args.load_static_path, pre_freeze_params)
    gf_data = get_gradually_freeze_data_2(cmd_args.load_gf_path, static_data)
    all_data = static_data + gf_data
    new_all_data = calc_speedup(all_data)
    print(len(all_data))

    hyper_params = cmd_args.load_gf_path.split('/')[-5]
    static_dt = cmd_args.load_static_path.split('/')[-2]
    gf_dt = cmd_args.load_gf_path.split('/')[-2]
    print(f'{hyper_params} {static_dt} {gf_dt}')
    
    

    # Create output folder
    script_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = f'./save/merge_plot_result/{script_time}__{hyper_params}__static={static_dt}__gf={gf_dt}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(new_all_data[0].keys())
    merged_csv_filename = os.path.join(output_dir, f'{hyper_params}__static={static_dt}__gf={gf_dt}__merged.csv')
    print(merged_csv_filename)
    utils.csv_exporter.export_csv(data=new_all_data, filepath=merged_csv_filename, fields=new_all_data[1].keys())


    plot_epoch_to_accuracy(all_data=new_all_data, output_dir=output_dir, timestamp=script_time, model_type=model_type)
    plot_duration(all_data=new_all_data, output_dir=output_dir, model_type=model_type)
    plot_volume(all_data=new_all_data, output_dir=output_dir, model_type=model_type)


    print(os.path.abspath(merged_csv_filename))
    

