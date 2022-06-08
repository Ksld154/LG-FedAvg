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


def calc_speedup(all_data):
    baseline_time = 0
    for idx,  data in enumerate(all_data):
        # print(data.keys())
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

def plot_speedup(all_data, output_dir, model_type):
    utils.myplotter.plot_speedup_ratio(all_data, title=f'Training Speedup (Model: {model_type})', figure_idx=6)
    utils.myplotter.save_figure(base_dir=output_dir, filename=f'{model_type}_speedup.png') 

def plot_trans_to_acc(all_data, output_dir, model_type):
    utils.myplotter.plot_trans_to_acc(all_data, title=f'Training Accuracy w.r.t.Transmission traffic volume (Model: {model_type})', figure_idx=4)
    utils.myplotter.save_figure(base_dir=output_dir, filename=f'{model_type}_volume_to_acc.png') 

def scatter_trans_to_bestacc(all_data, output_dir, model_type):
    utils.myplotter.scatter_trans_to_bestacc(all_data, title=f'Best Accuracy w.r.t.Transmission traffic volume (Model: {model_type})', figure_idx=5)
    utils.myplotter.save_figure(base_dir=output_dir, filename=f'{model_type}_volume_to_bestacc.png') 

if __name__ == '__main__':
    cmd_args = opt_parser()
    if not cmd_args.load_static_path or not cmd_args.load_gf_path:
        print('PATH ERROR')
        sys.exit(0)
         
    static_data = get_static_freeze_data(cmd_args.load_static_path)
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

    if 'resnet' in cmd_args.load_static_path:
        model_type = 'ResNet-18'
    elif 'mobilenet' in cmd_args.load_static_path:
        model_type = 'MobileNet'
    else:
        model_type = 'LeNet-5'
    
    print(new_all_data[0].keys())
    merged_csv_filename = os.path.join(output_dir, f'{hyper_params}__static={static_dt}__gf={gf_dt}__merged.csv')
    print(merged_csv_filename)
    utils.csv_exporter.export_csv(data=new_all_data, filepath=merged_csv_filename, fields=new_all_data[1].keys())

    scatter_trans_to_bestacc(all_data=new_all_data, output_dir=output_dir, model_type=model_type)
    plot_trans_to_acc(all_data=new_all_data, output_dir=output_dir, model_type=model_type)
    plot_accuracy_to_time_figure(all_data=new_all_data, output_dir=output_dir, timestamp=script_time, model_type=model_type)
    plot_best_acc(all_data=new_all_data, output_dir=output_dir, model_type=model_type)
    plot_transmission_ratio(all_data=new_all_data, output_dir=output_dir, model_type=model_type)
    plot_speedup(all_data=new_all_data, output_dir=output_dir, model_type=model_type)
    print(os.path.abspath(merged_csv_filename))
    
    # plot_accuracy_to_time_figure()
    # new_all_data = calc_transmission_speedup(all_data)
    # plot_transmission_speedup_figure(all_data=new_all_data, output_dir=output_dir, timestamp=script_time, model_type=model_type)
    # plot_transmission_speedup_figure(all_data=new_all_data, model_type=model_type)

