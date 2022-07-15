import os
import datetime

import utils.csv_exporter
import utils.myplotter
import utils.no_opportunistic_plotter
from constants import *


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
    utils.no_opportunistic_plotter.multiplot(
        all_data=all_data,
        y_label="Accuracy",
        title="",
        figure_idx=1
    )
    utils.no_opportunistic_plotter.legend()
    utils.no_opportunistic_plotter.save_figure(output_dir, f"{timestamp}_{model_type}_FL_epoch_to_Accuracy.eps")
    utils.no_opportunistic_plotter.save_figure(output_dir, f"{timestamp}_{model_type}_FL_epoch_to_Accuracy.png")



def plot_transmission_ratio(all_data, output_dir, model_type):
    utils.no_opportunistic_plotter.plot_transmission_ratio(all_data=all_data, title=f'', figure_idx=3)
    utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_transmission_volume_reduction.eps')
    utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_transmission_volume_reduction.png') 



def plot_speedup(all_data, output_dir, model_type):
    utils.no_opportunistic_plotter.plot_speedup_ratio(all_data, title=f'', figure_idx=4)
    utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_speedup.eps')
    utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_speedup.png') 



def plot_trans_to_acc(all_data, output_dir, model_type):
    utils.no_opportunistic_plotter.plot_trans_to_acc(all_data, title=f'', figure_idx=5)
    utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_volume_to_acc.eps')
    utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_volume_to_acc.png') 


def plot_epoch_to_trans(all_data, output_dir, timestamp, model_type):
    utils.no_opportunistic_plotter.plot_epoch_to_trans(
        all_data=all_data,
        y_label="Accuracy",
        title="",
        figure_idx=6
    )
    utils.no_opportunistic_plotter.save_figure(output_dir, f"{timestamp}_{model_type}_FL_epoch_to_trans.eps")
    utils.no_opportunistic_plotter.save_figure(output_dir, f"{timestamp}_{model_type}_FL_epoch_to_trans.png")


def plot_trans_to_target_acc(all_data, output_dir, model_type):
    utils.no_opportunistic_plotter.plot_trans_to_target_acc(all_data, title=f'', figure_idx=10, model_type=model_type)
    utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_volume_to_target_acc.eps')
    utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_volume_to_target_acc.png')

def plot_best_acc(all_data, output_dir, model_type):
    utils.no_opportunistic_plotter.plot_best_acc(all_data, title=f'', figure_idx=11)
    utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_best_acc.eps')
    utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_best_acc.png')
    utils.no_opportunistic_plotter.block_show()

def plot_total_transmission(all_data, output_dir, model_type):
    utils.no_opportunistic_plotter.plot_total_transmission(all_data, title=f'', figure_idx=13)
    utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_total_transmission.eps')
    utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'{model_type}_total_transmission.png')
    utils.no_opportunistic_plotter.block_show()

if __name__ == '__main__':
    
    model_type = 'MobileNet'
    pre_freeze_params = MOBILENET_PARAMS


    all_data = utils.csv_exporter.import_csv(filepath='mobilenet_no_opportunistic_switch.csv')
    print(f'{"name", "total_time", "transmission_time", "transmission_volume", "transmission_volume_readable"}')
    for row in all_data:
        print(f'{row["name"]} {row["total_time"]},{row["transmission_time"]}, {row["transmission_volume"]}, {row["transmission_volume_readable"]}')

    # Create output folder
    script_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = f'./save/no_opportunistic_switch_result/{script_time}_{model_type}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    plot_trans_to_target_acc(all_data=all_data, output_dir=output_dir, model_type=model_type)
    # exit(0)

    # plot_epoch_to_trans(all_data=all_data, output_dir=output_dir, timestamp=script_time, model_type=model_type)
    # plot_epoch_to_accuracy(all_data=all_data, output_dir=output_dir, timestamp=script_time, model_type=model_type)
    plot_best_acc(all_data=all_data, output_dir=output_dir, model_type=model_type)
    plot_total_transmission(all_data=all_data, output_dir=output_dir, model_type=model_type)
    plot_trans_to_target_acc(all_data=all_data, output_dir=output_dir, model_type=model_type)
    # plot_transmission_ratio(all_data=all_data, output_dir=output_dir, model_type=model_type)
    # plot_speedup(all_data=all_data, output_dir=output_dir, model_type=model_type)
    # plot_trans_to_acc(all_data=all_data, output_dir=output_dir, model_type=model_type)
    