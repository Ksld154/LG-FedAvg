import argparse
import sys
import os
import datetime

import utils.csv_exporter
import utils.group_plotter
from constants import *

def opt_parser():
    usage = 'Merge FL training results and plot figure from static freezing results and Gradually Freezing results.'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--load_group1', type=str, default='', help='Load Static Freeze results from path')
    parser.add_argument('--load_group2', type=str, default='', help='Load Gradually Freeze results from path')
    return parser.parse_args()


if __name__ == '__main__':
    cmd_args = opt_parser()
    if not cmd_args.load_group1 or not cmd_args.load_group2:
        print('PATH ERROR')
        sys.exit(0)
         
    gr1_data = utils.csv_exporter.import_csv(cmd_args.load_group1)
    gr2_data = utils.csv_exporter.import_csv(cmd_args.load_group2)
    all_data = [gr1_data, gr2_data]
    print(len(all_data))

    if 'resnet' in cmd_args.load_group1:
        model_type = 'ResNet-18'
    elif 'mobilenet' in cmd_args.load_group1:
        model_type = 'MobileNet'
    else:
        model_type = 'LeNet-5'
    

    # Create output folder
    script_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = f'./save/group_plot_result/{script_time}_{model_type}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    utils.group_plotter.plot_transmission_volume(all_data=all_data, title='', figure_idx=1)
    utils.group_plotter.save_figure(os.path.join(output_dir, f'{model_type}_{script_time}_trans.png'))
    utils.group_plotter.save_figure(os.path.join(output_dir, f'{model_type}_{script_time}_trans.eps'))


