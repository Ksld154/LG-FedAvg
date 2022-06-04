import argparse
import sys
import os
import datetime

import utils.csv_exporter
import utils.myplotter
from constants import *

def opt_parser():
    usage = 'Merge FL training results and plot figure from static freezing results and Gradually Freezing results.'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--load_static_path', type=str, default='', help='Load Static Freeze results from path')
    parser.add_argument('--load_gf_path', type=str, default='', help='Load Gradually Freeze results from path')
    
    return parser.parse_args()


def get_static_freeze_data(filepath):
    data = utils.csv_exporter.import_csv(filepath=filepath)
    return data

def get_gradually_freeze_data(filepath, static_freeze_data):
    pretrained_acc = static_freeze_data[0]['acc'][:WARM_UP_ROUNDS]
    print(pretrained_acc)
    
    data = utils.csv_exporter.import_simple_csv(filepath=filepath)
    new_data = []
    for row in data:
        row = [float(x) / 100.0  for x in row]
        acc = pretrained_acc + row
        new_data.append(dict(name="Gradually Freezing: Primary Model", acc=acc))
    # print(new_data)
    return new_data

def get_gradually_freeze_data_2(filepath, static_freeze_data):
    pretrained_acc = static_freeze_data[0]['acc'][:WARM_UP_ROUNDS]
    print(pretrained_acc)
    
    data = utils.csv_exporter.import_csv(filepath=filepath)
    print(data)
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
                                transmission_volume_readable=row['transmission_volume_readable']))
    print(new_data)
    return new_data

def plot_accuracy_to_time_figure(all_data, output_dir, timestamp, model_type):
    utils.myplotter.multiplot(
        all_data=all_data,
        y_label="Accuracy",
        title=f'FL Gradually Freezing Accuracy ({model_type})',
        figure_idx=1
    )

    utils.myplotter.legend()
    utils.myplotter.save_figure(output_dir, f"{timestamp}_{model_type}_FL_time_to_Accuracy.png")
    # utils.myplotter.show()


def calc_transmission_speedup(all_data):
    baseline_time = 0
    result_all_data = []
    for idx,  data in enumerate(all_data):
        if data.get('total_time'):
            pt = datetime.datetime.strptime(data['total_time'],'%H:%M:%S.%f')
            tt = pt - datetime.datetime(1900, 1, 1)
            total_seconds = tt.total_seconds()
            print(total_seconds)
            data['total_time'] = total_seconds
            if idx == 0:
                baseline_time = total_seconds
            
            data['speedup_ratio'] = 1 - (total_seconds/baseline_time)

        result_all_data.append(data)
    # print(result_all_data)
    return result_all_data

def plot_transmission_speedup_figure(all_data, output_dir, timestamp, model_type):
    utils.myplotter.plot_transmission_ratio(
        all_data=all_data,
        title= f'FL Gradually Freezing Speedup ({model_type})',
        figure_idx=1
    )

    # utils.myplotter.legend()
    utils.myplotter.save_figure(output_dir, f"{timestamp}_FL_Static_Freezing_Accuracy.png")
    utils.myplotter.show()


def plot_best_acc(all_data, output_dir, model_type):
    utils.myplotter.plot_best_acc(all_data=all_data, title=f'Best Accuracy w.r.t.Initial Freezing Point (Model: {model_type})', figure_idx=2)
    utils.myplotter.save_figure(base_dir=output_dir, filename=f'{model_type}_best_acc.png') 

def plot_transmission_ratio(all_data, output_dir, model_type):
    utils.myplotter.plot_transmission_ratio(all_data=all_data, title=f'Transmission volume reduction w.r.t.Initial Freezing Point (Model: {model_type})', figure_idx=3)
    utils.myplotter.save_figure(base_dir=output_dir, filename=f'{model_type}_transmission_volume_reduction.png') 


if __name__ == '__main__':
    cmd_args = opt_parser()
    if not cmd_args.load_static_path or not cmd_args.load_gf_path:
        print('PATH ERROR')
        sys.exit(0)
         
    static_data = get_static_freeze_data(cmd_args.load_static_path)
    gf_data = get_gradually_freeze_data_2(cmd_args.load_gf_path, static_data)
    all_data = static_data + gf_data
    print(len(all_data))

    script_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = f'./save/merge_plot_result/{script_time}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if 'resnet' in cmd_args.load_static_path:
        model_type = 'ResNet-18'
    elif 'mobilenet' in cmd_args.load_static_path:
        model_type = 'MobileNet'
    else:
        model_type = 'LeNet-5'

    # plot_accuracy_to_time_figure(all_data=all_data, output_dir=output_dir, timestamp=script_time, model_type=model_type)
    plot_accuracy_to_time_figure(all_data=all_data, output_dir=output_dir, timestamp=script_time, model_type=model_type)
    plot_best_acc(all_data=all_data, output_dir=output_dir, model_type=model_type)
    plot_transmission_ratio(all_data=all_data, output_dir=output_dir, model_type=model_type)
    
    # plot_accuracy_to_time_figure()
    # new_all_data = calc_transmission_speedup(all_data)
    # plot_transmission_speedup_figure(all_data=new_all_data, output_dir=output_dir, timestamp=script_time, model_type=model_type)
    # plot_transmission_speedup_figure(all_data=new_all_data, model_type=model_type)

