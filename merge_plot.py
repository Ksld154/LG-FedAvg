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


def plot_figure(all_data, output_dir, timestamp):
    utils.myplotter.multiplot(
        all_data=all_data,
        y_label="Accuracy",
        title='FL Gradually Freezing Accuracy (LeNet-5)',
        figure_idx=1
    )

    utils.myplotter.legend()
    utils.myplotter.save_figure(output_dir, f"{timestamp}_FL_Static_Freezing_Accuracy.png")
    utils.myplotter.show()


if __name__ == '__main__':
    cmd_args = opt_parser()
    if not cmd_args.load_static_path or not cmd_args.load_gf_path:
        print('PATH ERROR')
        sys.exit(0)
         
    static_data = get_static_freeze_data(cmd_args.load_static_path)
    gf_data = get_gradually_freeze_data(cmd_args.load_gf_path, static_data)
    all_data = static_data + gf_data
    print(len(all_data))

    script_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = f'./save/merge_plot_result/{script_time}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plot_figure(all_data=all_data, output_dir=output_dir, timestamp=script_time)
