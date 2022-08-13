#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import datetime
import imp
import pickle
import time
from tracemalloc import start
import numpy as np
import pandas as pd
import torch
from constants import WARM_UP_ROUNDS

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from utils.tools import convert_size
from models.Update import LocalUpdate
from models.test import test_img
import os
import utils.myplotter
import utils.csv_exporter

import pdb
import torchinfo

class Experiment():
    def __init__(self, args) -> None:
        self.args = args
        self.results = []
        
        self.results_dir = None
        self.base_filename = ''
    
    def pre_train(self, net_glob, output_dir):
        # results_save_path = os.path.join(self.base_dir, 'fed/results.csv')

        loss_train = []
        best_acc = None
        net_best = None
        best_loss = None
        best_epoch = None

        lr = self.args.lr
        results = []
        

        for iter in range(args.pre_trained_rounds):
            w_glob = None
            loss_locals = []
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
            print(f'Round {(iter+1):3d}, lr: {lr:.6f}, {idxs_users}')


            for idx in idxs_users:
                local = LocalUpdate(args=self.args, dataset=dataset_train, idxs=dict_users_train[idx])
                net_local = copy.deepcopy(net_glob)

                w_local, loss = local.train(net=net_local.to(self.args.device))
                loss_locals.append(copy.deepcopy(loss))

                if w_glob is None:
                    w_glob = copy.deepcopy(w_local)
                else:
                    for k in w_glob.keys():
                        w_glob[k] += w_local[k]

            lr *= self.args.lr_decay

            # update global weights
            for k in w_glob.keys():
                w_glob[k] = torch.div(w_glob[k], m)
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)

            if (iter + 1) % self.args.test_freq == 0:
                net_glob.eval()
                acc_test, loss_test = test_img(net_glob, dataset_test, args)
                print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                    iter+1, loss_avg, loss_test, acc_test))


                if best_acc is None or acc_test > best_acc:
                    best_acc = acc_test
                    net_best = copy.deepcopy(net_glob)
                    best_epoch = iter+1

                results.append(np.array([iter+1, loss_avg, loss_test, acc_test, best_acc]))
                final_results = np.array(results)
                final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
                # final_results.to_csv(results_save_path, index=False)

        self.results = results

        pre_trained_model_path = os.path.join(output_dir, "pretrained.pt")
        print(pre_trained_model_path)
        torch.save(net_glob.state_dict(), pre_trained_model_path)

        return net_glob, results


    def plot_figure(self, all_results, output_dir):
        utils.myplotter.multiplot(
                all_data = all_results, 
                y_label="Accuracy",
                title="FL Static Freezing Accuracy", 
                figure_idx=1
            )

        utils.myplotter.legend()
        utils.myplotter.save_figure(output_dir, f"{self.base_filename}_FL_Static_Freezing_Accuracy.png")
        utils.myplotter.show()
    
    def output_csv(self, data, output_dir, fields):
        
        csv_file = os.path.join(output_dir, f"{self.base_filename}_result.csv")
        print(csv_file)
        utils.csv_exporter.export_csv(data=data, filepath=csv_file, fields=fields)


class StaticFreeze():
    def __init__(self, args, name, results, base_dir) -> None:
        self.args = args
        self.name = name
        self.results = results
        self.base_dir = base_dir
        
        self.best_acc = None
        self.best_epoch = None
        self.net_best = None

        self.final_results = None
        self.total_time = datetime.timedelta(seconds=0)
        self.transmission_time = datetime.timedelta(seconds=0)
        self.transmission_volume = 0
        self.transmission_volume_history = []

        self.total_time_history = []
        self.transmission_time_history = []
        
        

    # This is for Static Freezing Training !!!
    def train(self, net_glob, freeze_degree):
        # training
        results_save_path = os.path.join(self.base_dir, 'fed/results.csv')

        loss_train = []
        lr = self.args.lr
        results = []
        final_results = None
        total_time = datetime.timedelta(seconds=0)


        # [Experiment #1] Static Freeze global model
        for idx, l in enumerate(net_glob.layers):
            if (idx+1) <= freeze_degree:
                l.requires_grad_(False)
        torchinfo.summary(net_glob, (1,3,32,32), device=self.args.device)
        

        for iter in range(self.args.epochs-args.pre_trained_rounds):
            epoch = iter+args.pre_trained_rounds
            
            w_glob = None
            loss_locals = []
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
            print(f'Round {(epoch+1):3d}, lr: {lr:.6f}, {idxs_users}')

            max_upload_time = datetime.timedelta(seconds=0)
            max_download_time = datetime.timedelta(seconds=0)
            max_local_train_time = datetime.timedelta(seconds=0)

            for idx in idxs_users:
                local = LocalUpdate(args=self.args, dataset=dataset_train, idxs=dict_users_train[idx])
                net_local = copy.deepcopy(net_glob)

                w_local, loss = local.train(net=net_local.to(self.args.device))
                loss_locals.append(copy.deepcopy(loss))

                if w_glob is None:
                    w_glob = copy.deepcopy(w_local)
                else:
                    for k in w_glob.keys():
                        w_glob[k] += w_local[k]

                local.calc_train_and_transmission_time(active_workers=len(idxs_users))
                max_local_train_time = max(local.local_train_time, max_local_train_time)
                max_upload_time = max(local.upload_time, max_upload_time)
                max_download_time = max(local.download_time, max_download_time)
                
                self.transmission_volume += local.trainable_params*4*8  # in bits
            
            iteration_round_time = max_local_train_time + max_upload_time + max_download_time
            total_time += iteration_round_time
            print(f'Round {(epoch+1):3d}, current round duration: {iteration_round_time}')
            print(f'Round {(epoch+1):3d}, cumulated transmission: {convert_size(self.transmission_volume)}')
            print(f'Round {(epoch+1):3d}, transmission:{max_upload_time + max_download_time}')
            print(f'Round {(epoch+1):3d}, local train :{max_local_train_time}')
            
            self.transmission_time +=  max_upload_time + max_download_time
            self.transmission_volume_history.append(self.transmission_volume)
            
            self.transmission_time_history.append(self.transmission_time)
            self.total_time_history.append(self.total_time)

            lr *= self.args.lr_decay

            # update global weights
            for k in w_glob.keys():
                w_glob[k] = torch.div(w_glob[k], m)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)

            if (epoch + 1) % self.args.test_freq == 0:
                net_glob.eval()
                acc_test, loss_test = test_img(net_glob, dataset_test, args)
                print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                    epoch+1, loss_avg, loss_test, acc_test))


                if self.best_acc is None or acc_test > self.best_acc:
                    self.net_best = copy.deepcopy(net_glob)
                    self.best_acc = acc_test
                    self.best_epoch = epoch+1

                results.append(np.array([epoch+1, loss_avg, loss_test, acc_test, self.best_acc]))
                self.results.append(np.array([epoch+1, loss_avg, loss_test, acc_test, self.best_acc]))

            if (epoch + 1) % 50 == 0:
                best_save_path = os.path.join(self.base_dir, 'fed/best_{}.pt'.format(epoch + 1))
                model_save_path = os.path.join(self.base_dir, 'fed/model_{}.pt'.format(epoch + 1))
                torch.save(self.net_best.state_dict(), best_save_path)
                torch.save(net_glob.state_dict(), model_save_path)
        
        print(f'Total duration: {total_time}')

        final_results = np.array(self.results)
        final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
        final_results.to_csv(results_save_path, index=False)

        self.final_results = final_results
        self.total_time = total_time



if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    start_time = time.time()
    current_time = datetime.datetime.now()
    script_time = current_time.strftime("%Y-%m-%d_%H%M%S")
    print(f'Training Start: {current_time}')

    result_dir = f'./save/{args.dataset}/{args.model}_iid{args.iid}_num{args.num_users}_C{args.frac}_le{args.local_ep}/shard{args.shard_per_user}/{args.results_save}/{script_time}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    
    # build model
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    net_glob = get_model(args)
    net_glob.train()
    
    exp1 = Experiment(args=args)
    exp1.base_filename = f"{args.model}_e={args.epochs}_window={args.window_size}_static"
    print(exp1.base_filename)
    pretrained_net_glob, pre_trained_results = exp1.pre_train(net_glob, result_dir)
    
    
    all_results = []
    all_accs = []
    
    if args.model == 'mobilenet':
        degree_step = 3
    elif args.model == 'resnet':
        degree_step = 2
    else:
        degree_step = 1

    for d in range(args.static_freeze_candidates):
        degree = d * degree_step
        
        base_dir = f'./save/{args.dataset}/{args.model}_iid{args.iid}_num{args.num_users}_C{args.frac}_le{args.local_ep}/shard{args.shard_per_user}/{args.results_save}/{script_time}/static_{degree}/'
        print(base_dir)
        if not os.path.exists(os.path.join(base_dir, 'fed')):
            os.makedirs(os.path.join(base_dir, 'fed'), exist_ok=True)
        
        static_freeze_exp = StaticFreeze(args=args, name=f'Static Freeze: {degree} layers', results=copy.deepcopy(pre_trained_results), base_dir=base_dir)
        if degree == 0:
            static_freeze_exp.name = "Baseline: No Freeze"
        static_freeze_exp.train(net_glob=copy.deepcopy(pretrained_net_glob), freeze_degree=degree)
        # all_results.append(static_freeze_exp.results)


        print(f'Static Freeze Degree: {degree}')
        print('Best model, iter: {}, acc: {}'.format(static_freeze_exp.best_epoch, static_freeze_exp.best_acc))
        accuracy = static_freeze_exp.final_results.acc_test.to_numpy()
        acc_list = list(accuracy)
        acc_list = [x / 100.0  for x in acc_list]

        print(acc_list)
        print(np.array2string(accuracy, separator=', '))
        all_accs.append(accuracy)
        all_results.append(
            dict(name=static_freeze_exp.name, 
                    acc=acc_list, 
                    total_time=static_freeze_exp.total_time, 
                    transmission_time=static_freeze_exp.transmission_time, 
                    transmission_volume=static_freeze_exp.transmission_volume,
                    transmission_volume_readable=convert_size(static_freeze_exp.transmission_volume),
                    transmission_volume_history=static_freeze_exp.transmission_volume_history,
                    total_time_history=static_freeze_exp.total_time_history,
                    transmission_time_history=static_freeze_exp.transmission_time_history))

    
    print(all_results)
    print(all_accs)

    end_time = time.time()
    print(f'Total training time: {datetime.timedelta(seconds= end_time - start_time)}')


    exp1.output_csv(data=all_results, output_dir=result_dir, fields=['name', 'acc', 'total_time', 'transmission_time', 'transmission_volume', 'transmission_volume_readable', 'transmission_volume_history', 'total_time_history', 'transmission_time_history'])
    exp1.plot_figure(all_results=all_results, output_dir=result_dir)