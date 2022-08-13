#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
from email.mime import base
from json import load
import pickle
from unittest import mock
import numpy as np
import pandas as pd
import torch
import datetime
import time
import torchinfo
import numpy as np

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img
import os
import json

# my library
from models.trainer import GlobalTrainer, LocalTrainer, MyModel
from utils.tools import moving_average, convert_size
import utils.myplotter as myplotter
import utils.csv_exporter as csv_exporter
from constants import *

# import pdb

class Experiment():
    def export_cmd_args(self, cmd_args, filepath):
        print(cmd_args)
        csv_exporter.export_csv(data=[cmd_args], filepath=f'{filepath}.csv', fields=cmd_args.keys())
        
        with open(f'{filepath}.json', 'w') as fp:
            json.dump(cmd_args, fp, indent=4, sort_keys=True)


    def plot_figure(self, all_results, output_dir):
        myplotter.multiplot(
                all_data = all_results, 
                y_label="Accuracy",
                title="FL Gradually Freezing Accuracy", 
                figure_idx=1
            )

        myplotter.legend()
        myplotter.save_figure(output_dir, "FL_Gradually_Freezing_Accuracy.png")
        myplotter.show()
    
    def output_csv(self, data, output_dir, fields):
        csv_file = os.path.join(output_dir, "result.csv")
        print(csv_file)
        csv_exporter.export_csv(data=data, filepath=csv_file, fields=fields)

if __name__ == '__main__':
    # parse args
    args = args_parser()


    start_time = time.time()
    current_time = datetime.datetime.now()
    print(f'Training Start: {current_time}')
    script_time = current_time.strftime("%Y-%m-%d_%H%M%S")


    base_dir = f'./save/{args.dataset}/{args.model}_iid{args.iid}_num{args.num_users}_C{args.frac}_le{args.local_ep}/shard{args.shard_per_user}/{args.results_save}/{script_time}/'
    print(base_dir)
    if not os.path.exists(os.path.join(base_dir, 'fed')):
        os.makedirs(os.path.join(base_dir, 'fed'), exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    dict_save_path = os.path.join(base_dir, 'dict_users.pkl')
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)
    
    exp1 = Experiment()
    cmd_args_path = os.path.join(base_dir, "cmd_args")
    exp1.export_cmd_args(cmd_args=vars(args), filepath=cmd_args_path)
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f'GPU ID: {args.device}')
    # exit(0)

    # build model
    net_glob = get_model(args)
    if args.load_pretrained:
        pretrained_path = args.load_pretrained
        print(pretrained_path)
        if os.path.exists(pretrained_path):
            print(f'Use pretrained model: {pretrained_path}')
            net_glob.load_state_dict(torch.load(pretrained_path))
            args.epochs -= args.pre_trained_rounds
            
        else:
            print(f'[ERROR] Pretrain model not exists!')
            print(f'{pretrained_path}')

    
    net_glob.train()
    net_glob_second = copy.deepcopy(net_glob)


    g_trainer = GlobalTrainer(args= args, net=net_glob, net_secondary=net_glob_second)

    # setup each freezing degree model for brute force search
    for idx in range(5):
        g_trainer.brute_force_nets[idx] = MyModel(model=copy.deepcopy(g_trainer.net.model), args=args, freeze_degree=idx+1) 

    # training
    results_save_path = os.path.join(base_dir, 'fed/results.csv')
    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []
    
    ### my parameters ###
    mock_gradually_freezing_degree = 0 # for static freezing
    freeze_degree = 0 # global record the freezing degree for local side
    switch_model_flag = False
    window_size_cnt = 0
    print(args.switch_model)
    total_time = datetime.timedelta(seconds=0)
    total_trainable_params = 0




    all_local_trainers = []
    for idx in range(args.num_users):
        local_trainer = LocalTrainer(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
        all_local_trainers.append(local_trainer)


    for e in range(args.epochs):
        g_trainer.weights = None
        lr *= args.lr_decay

        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)  # number of workers per round
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f'Round {(e+1):3d}, lr: {lr:.6f}, {idxs_users}')


        for idx in idxs_users:
            # local_trainer = LocalTrainer(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            local_trainer = all_local_trainers[idx]

            # Train local primary model
            local_primary_model = copy.deepcopy(g_trainer.net.model)
            local_primary_model.train()
            local_trainer.net_primary = MyModel(model=local_primary_model, args=args, freeze_degree=g_trainer.net.freeze_degree)
            w_local, loss = local_trainer.train(net=local_trainer.net_primary.model.to(args.device), lr=lr)

            loss_locals.append(copy.deepcopy(loss))
            local_trainer.net_primary.update_loss_train_delta(loss=loss)
            local_trainer.net_primary.loss_train.append(loss)

            # increament global weights (aggregation)
            if g_trainer.weights is None:
                g_trainer.weights = copy.deepcopy(w_local)
            else: 
                for k in g_trainer.weights.keys():
                    g_trainer.weights[k] += w_local[k]
            
            local_trainer.calc_train_and_transmission_time(active_workers= len(idxs_users))
            total_trainable_params += local_trainer.trainable_params
            print(local_trainer.trainable_params)

        
        active_local_trainers = [all_local_trainers[i] for i in idxs_users]
        max_local_train_time = max(local_trainer.local_train_time for local_trainer in active_local_trainers)
        max_upload_time = max(local_trainer.upload_time for local_trainer in active_local_trainers)
        max_download_time = max(local_trainer.download_time for local_trainer in active_local_trainers)
        
        iteration_round_time = max_local_train_time + max_upload_time + max_download_time
        total_time += iteration_round_time
        g_trainer.transmission_time += max_upload_time + max_download_time
        print(f'Round {(e+1):3d}, time elapsed:{iteration_round_time}')
        print(f'Round {(e+1):3d}, transmission:{max_upload_time + max_download_time}')
        print(f'Round {(e+1):3d}, local train :{max_local_train_time}')

       
        g_trainer.transmission_volume = total_trainable_params * 4 * 8
        g_trainer.transmission_volume_history.append(g_trainer.transmission_volume)
        print(f'Round {(e+1):3d}, cumulated transmission: {convert_size(g_trainer.transmission_volume)}')
        
        g_trainer.transmission_time_history.append(g_trainer.transmission_time)
        g_trainer.total_time_history.append(total_time)

        # global primary_model aggregation
        for k in g_trainer.weights.keys():
            g_trainer.weights[k] = torch.div(g_trainer.weights[k], m)  # new weights (weights after aggregation)


        # New Aggregation for primary model
        old_weights = copy.deepcopy(g_trainer.net.model.state_dict())  # weights before aggregation?

        for idx, k in enumerate(g_trainer.weights.keys()):
            # frozen layers should use old weights
            if any(substr in k for substr in g_trainer.net.frozen_layers_name):
                # print(f'{g_trainer.net.freeze_degree} {k}')
                g_trainer.weights[k] = copy.deepcopy(old_weights[k]) 
        g_trainer.net.model.load_state_dict(g_trainer.weights)


       
        # exit(0)
       
        # [FL-8] FL Generate Secondary Model Method#2 global secondary_model aggregation
        # g_trainer.generate_secondary_model_method_1(old_primary_weights=copy.deepcopy(old_weights), epoch=e)
        g_trainer.new_generate_secondary_model_method_1(old_primary_weights=copy.deepcopy(old_weights), epoch=e)

        # g_trainer.generate_secondary_model_method_2(epoch=e)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        g_trainer.net.update_loss_train_delta(loss=loss_avg)
        g_trainer.net.loss_train.append(loss_avg)
        g_trainer.net_secondary.update_loss_train_delta(loss=loss_avg)
        g_trainer.net_secondary.loss_train.append(loss_avg)


        # test/validate global model
        acc_test = loss_test = None
        if (e+1) % args.test_freq == 0:
            g_trainer.net.model.eval()
            acc_test, loss_test = test_img(g_trainer.net.model, dataset_test, args)
            print(f'Round {(e+1):3d}, Average loss {loss_avg:.3f}, Test loss {loss_test:.3f}, Test accuracy: {acc_test:.2f} [Primary   model: {g_trainer.net.freeze_degree}]')

            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(g_trainer.net.model)
                best_acc = acc_test
                best_epoch = e

            results.append(np.array([e, loss_avg, loss_test, acc_test, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
            final_results.to_csv(results_save_path, index=False)

            g_trainer.net.update_loss_test_delta(loss=loss_test)
            g_trainer.net.loss_test.append(loss_test)
            g_trainer.net.acc.append(acc_test)

            
            # test secondary global model here
            g_trainer.net_secondary.model.eval()
            acc_test_2, loss_test_2 = test_img(g_trainer.net_secondary.model, dataset_test, args)
            print(f'Round {(e+1):3d}, Average loss {loss_avg:.3f}, Test loss {loss_test_2:.3f}, Test accuracy: {acc_test_2:.2f} [Secondary model: {g_trainer.net_secondary.freeze_degree}]')
            g_trainer.net_secondary.update_loss_test_delta(loss=loss_test_2)
            g_trainer.net_secondary.loss_test.append(loss_test_2)
            g_trainer.net_secondary.acc.append(acc_test_2)

            g_trainer.models_loss_test_diff.append(loss_test_2 - loss_test)
            g_trainer.models_loss_test_diff_ratio.append((loss_test_2 - loss_test) / loss_test)

        if (e+1) % 50 == 0:
            best_save_path = os.path.join(base_dir, f'fed/best_{(e + 1)}.pt')
            model_save_path = os.path.join(base_dir, f'fed/model_{(e + 1)}.pt')
            torch.save(net_best.state_dict(), best_save_path)
            torch.save(g_trainer.net.model.state_dict(), model_save_path)
        
        if (e+1) % 10 == 0:
            now = time.time()
            print(f'Round {(e+1):3d}, Current Time: {datetime.datetime.now()}')
            print(f'Elapsed Time: {datetime.timedelta(seconds= now - start_time)}')

        # [Experiment \#2] Gradually freezing(decision at global side, freezing also at global side)
        
        # Switch model decision
        window_size_cnt += 1        
        avg_loss_diff = moving_average(g_trainer.models_loss_test_diff, args.window_size)
        avg_loss_diff_ratio = moving_average(g_trainer.models_loss_test_diff_ratio, args.window_size)
        print(f'Round {(e+1):3d}, Average Model Loss Difference: {avg_loss_diff}')
        print(f'Round {(e+1):3d}, Average Model Ratio Loss Difference: {avg_loss_diff_ratio} {g_trainer.models_loss_test_diff_ratio[-1]}')
        
        if not args.optimistic_train:
            print(f'Converged Threshold: {args.converged_threshold}')
            if args.switch_model and g_trainer.net.is_converged(args.converged_threshold) and window_size_cnt >= args.window_size:
                print("*** [No optimistic_train] Models is converged, let's goto next freezing degree! ***")
                g_trainer.further_freeze_without_opportunistic_train()
                window_size_cnt = 0

        


        if window_size_cnt >= args.window_size and not np.isnan(avg_loss_diff_ratio) and avg_loss_diff < args.loss_diff_ratio:
            print(f"Secondary model ratio is tolarably good: {avg_loss_diff_ratio}, let's switch model")
            g_trainer.switch_model()
            window_size_cnt = 0

    

    end_time = time.time()
    print(f'Best model, iter: {best_epoch}, acc: {best_acc}')
    print(f'Total training time: {datetime.timedelta(seconds= end_time - start_time)}')
    print(f'Total Time: {total_time}')

    accuracy = final_results.acc_test.to_numpy()


    print(g_trainer.net.loss_train)
    print(g_trainer.net.loss_test)
    print(g_trainer.net.loss_train_delta)
    print(g_trainer.net.loss_test_delta)
    print(g_trainer.net.acc)
    print(g_trainer.net_secondary.acc)

    csv_data = [g_trainer.net.acc, g_trainer.net_secondary.acc]
    if args.brute_force:
        for idx, k in enumerate(g_trainer.brute_force_nets):
            print(k.acc)
            csv_data.append(k.acc)
    csv_exporter.export(base_dir, 'accuracy.csv', csv_data)

    # myplotter.setup_plot("Global Metrics of FL w/ LeNet-5 Model on CIFAR 10 Dataset", "Loss", 1)
    # myplotter.plot_data(g_trainer.net.loss_test, "Primary Test Loss")
    # myplotter.plot_data(g_trainer.net_secondary.loss_test, "Secondary Test Loss")
    # myplotter.plot_data(g_trainer.net.loss_train, "Primary Train Loss")
    # myplotter.legend()
    # myplotter.save_figure(base_dir, "global_model_metrics")
    
    # myplotter.setup_plot("Global Metrics of FL w/ LeNet-5 Model on CIFAR 10 Dataset", "Accuracy", 2)
    # myplotter.plot_data(g_trainer.net.acc, "Global Primary Model")
    # myplotter.plot_data(g_trainer.net_secondary.acc, "Global Secondary Model")
    # if args.brute_force:
    #     for idx, k in enumerate(g_trainer.brute_force_nets):
    #         myplotter.plot_data(k.acc, f"Brute-Force Freeze degree: {idx}")
    # myplotter.legend()
    # myplotter.save_figure(base_dir, "global_model_accuracy")
    
    # myplotter.show()


    print(np.array2string(accuracy, separator=', '))
    


    # export new version .csv file and plot .png file
    acc_list = list(accuracy)
    acc_list = [x / 100.0  for x in acc_list]
    all_results = []
    all_results.append(dict(name=f'Gradually Freeze: Primary Model', 
                            acc=acc_list, 
                            total_time=total_time, 
                            total_trainable_params=total_trainable_params,
                            transmission_time=g_trainer.transmission_time,
                            transmission_volume=g_trainer.transmission_volume,
                            transmission_volume_readable=convert_size(g_trainer.transmission_volume),
                            transmission_volume_history=g_trainer.transmission_volume_history,
                            total_time_history=g_trainer.total_time_history,
                            transmission_time_history=g_trainer.transmission_time_history))

    exp1.output_csv(data=all_results, output_dir=base_dir, fields=['name', 'acc', 'total_time', 'total_trainable_params', 'transmission_time', 'transmission_volume', 'transmission_volume_readable', 'transmission_volume_history', 'total_time_history', 'transmission_time_history'])
    exp1.plot_figure(all_results=all_results, output_dir=base_dir)