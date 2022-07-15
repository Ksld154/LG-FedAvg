import os
import datetime

from matplotlib.pyplot import plot

import utils.csv_exporter
import utils.myplotter
import utils.no_opportunistic_plotter
from constants import *



def plot_trans_to_target_acc_old(all_data, output_dir):
    # resnet_data = all_data[0]
    # mobilenet_data = all_data[1]

    # resnet_transmission = [eval(model['transmission_volume_history']) for model in resnet_data]
    # resnet_transmission_readable = [float(x)/8/1024/1024/1024 for x in resnet_transmission]

    # mobilenet_transmission = [eval(model['transmission_volume_history']) for model in mobilenet_data]
    # mobilenet_transmission_readable = [float(x)/8/1024/1024/1024 for x in mobilenet_transmission]
    

    # all_transmission_volume = [resnet_transmission_readable, mobilenet_transmission_readable]
    # print(all_transmission_volume)


    # for idx, model_group in enumerate(all_transmission_volume):
        
    #     for idx, model in enumerate(model_group):
    #         target_acc_idx = 0

    #         model_name.append(model['name'])
            
    #         transmission_volumes = eval(data['transmission_volume_history'])
    #         data_transmission_in_mb = [float(x)/8/1024/1024/1024 for x in transmission_volumes]
    #         print(len(data_transmission_in_mb))
    #         # print(data_transmission_in_mb)
    #         if 'No Freeze' in data["name"]:
    #             data['acc'] = data['acc'][WARM_UP_ROUNDS:]
            
    #         accs = data['acc']
    #         print(len(accs))

    #         target_trans_volume = []
    #         for idx, acc in enumerate(accs):
    #             if target_acc_idx < len(target_accs) and acc >= target_accs[target_acc_idx]:
    #                 target_acc_idx += 1
    #                 print(f'{acc}, {idx}')
    #                 target_trans_volume.append(data_transmission_in_mb[idx])

    #         print(target_trans_volume)

    #         for i in range(len(target_accs) - len(target_trans_volume)):
    #             target_trans_volume.append(0)
    #         all_group_data.append(target_trans_volume)    
    #     print(all_group_data)


    # utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'_volume_to_target_acc.eps')
    # utils.no_opportunistic_plotter.save_figure(base_dir=output_dir, filename=f'_volume_to_target_acc.png')
    pass


def plot_trans_to_target_acc_new(all_data, output_dir):
    utils.no_opportunistic_plotter.plot_trans_to_target_acc(all_data=all_data, figure_idx=19, model_type='merged', title='')
    utils.no_opportunistic_plotter.save_figure(base_dir='.', filename=f'merged_transmission_volume_to_target_acc.eps')
    utils.no_opportunistic_plotter.save_figure(base_dir='.', filename=f'merged_transmission_volume_to_target_acc.png')


def plot_accuracy_to_transmission_volume_ratio(all_data):
    # 1. get best acc
    resnet_data = all_data[0]
    mobilenet_data = all_data[1]

    resnet_best_accs = [max(model['acc']) for model in resnet_data]
    mobilenet_best_accs = [max(model['acc']) for model in mobilenet_data]
    all_best_accs = [resnet_best_accs, mobilenet_best_accs]
    print(all_best_accs)
    
    # 2. get total_transmission volume
    resnet_transmission_volume = [float(model['transmission_volume'])/8/1024/1024/1024 for model in resnet_data]
    mobilenet_transmission_volume = [float(model['transmission_volume'])/8/1024/1024/1024 for model in mobilenet_data]
    all_transmission_volume = [resnet_transmission_volume, mobilenet_transmission_volume]
    print(all_transmission_volume)

    # 3. calculate the ratio
    resnet_accuracy_to_transmission_volume_ratio = [ x/y for _, (x, y) in enumerate(zip(resnet_best_accs, resnet_transmission_volume)) ]
    mobilenet_accuracy_to_transmission_volume_ratio = [ x/y for _, (x, y) in enumerate(zip(mobilenet_best_accs, mobilenet_transmission_volume)) ]
    all_ratio = [resnet_accuracy_to_transmission_volume_ratio, mobilenet_accuracy_to_transmission_volume_ratio]
    print(all_ratio)

    # 4. plot
    model_name = [model['name'] for model in resnet_data]
    utils.no_opportunistic_plotter.plot_accuracy_to_transmission_volume_ratio(
        all_data=all_ratio, 
        figure_idx=15, 
        model_name=model_name)
    utils.no_opportunistic_plotter.save_figure(base_dir='.', filename=f'merged_accuracy_to_transmission_volume_ratio.eps')
    utils.no_opportunistic_plotter.save_figure(base_dir='.', filename=f'merged_accuracy_to_transmission_volume_ratio.png')
    # utils.no_opportunistic_plotter.block_show()

def plot_epoch_to_target_acc(all_data, output_dir):
    utils.no_opportunistic_plotter.plot_epoch_to_target_acc(all_data=all_data, figure_idx=20, model_type='merged', title='')
    utils.no_opportunistic_plotter.save_figure(base_dir='.', filename=f'merged_epoch_to_target_acc.eps')
    utils.no_opportunistic_plotter.save_figure(base_dir='.', filename=f'merged_epoch_to_target_acc.png')


if __name__ == '__main__':    

    # merge 2 kinds of model on the same graph
    resnet_data = utils.csv_exporter.import_csv(filepath='resnet_no_opportunistic_switch_2.csv')
    mobilenet_data = utils.csv_exporter.import_csv(filepath='mobilenet_no_opportunistic_switch.csv')
    plot_accuracy_to_transmission_volume_ratio(
        all_data=[resnet_data, mobilenet_data] 
    )

    
    merged_data = utils.csv_exporter.import_csv(filepath='merged_no-opp_switch.csv')
    plot_trans_to_target_acc_new(all_data=merged_data, output_dir='.')
    plot_epoch_to_target_acc(all_data=merged_data, output_dir='.')
