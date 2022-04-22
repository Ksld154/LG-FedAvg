#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
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

# my library
from models.trainer import GlobalTrainer, LocalTrainer, MyModel
from utils.tools import moving_average
import utils.myplotter as myplotter
from constants import *

import pdb

def decide_further_freeze():
    pass

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f'GPU ID: {args.device}')

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

    # build model
    net_glob = get_model(args)
    net_glob.train()
    net_glob_second = copy.deepcopy(net_glob)
    g_trainer = GlobalTrainer(args= args, net=net_glob, net_secondary=net_glob_second)

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
            local_trainer.net_primary = MyModel(model=local_primary_model, args=args, freeze_degree=0)
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

        # global primary_model aggregation
        for k in g_trainer.weights.keys():
            g_trainer.weights[k] = torch.div(g_trainer.weights[k], m)
        # g_trainer.net.model.load_state_dict(g_trainer.weights)
        
        # New Aggregation
        old_weights = g_trainer.net.model.state_dict() 
        for idx, k in enumerate(g_trainer.weights.keys()):
            # times 2 for weights and bias in single layer (LeNet-5)
            if idx < g_trainer.net.freeze_degree * 2:  # use old weights
                print(k)
                g_trainer.weights[k] = copy.deepcopy(old_weights[k])
                # print(g_trainer.weights[k])
        g_trainer.net.model.load_state_dict(g_trainer.weights)
        


        # global secondary_model aggregation
        old_secondary_weights = g_trainer.net_secondary.model.state_dict() 
        g_trainer.weights_secondary = copy.deepcopy(g_trainer.weights)   
        if (e+1) > WARM_UP_ROUNDS:
            for idx, k in enumerate(g_trainer.weights_secondary.keys()):
                # times 2 for weights and bias in single layer (LeNet-5)
                if idx < g_trainer.net_secondary.freeze_degree * 2:  # use old weights
                    print(k)
                    g_trainer.weights_secondary[k] = copy.deepcopy(old_secondary_weights[k])
        g_trainer.net_secondary.model.load_state_dict(g_trainer.weights_secondary)

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

            if (e+1) > WARM_UP_ROUNDS:
                # g_trainer.models_loss_test_diff.append(loss_test_2 - loss_test)
                g_trainer.models_loss_test_diff.append(abs(loss_test_2 - loss_test))
            


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
        # print(f'Global Primary   Model Converged: {g_trainer.net.is_converged()}')
        # print(f'Global Secondary Model Converged: {g_trainer.net_secondary.is_converged()}')
        
        # Switch model decision
        window_size_cnt += 1        
        avg_loss_diff = moving_average(g_trainer.models_loss_test_diff, args.window_size)
        print(f'Round {(e+1):3d}, Average Model Loss Difference: {avg_loss_diff}')
        if args.switch_model and g_trainer.net.is_converged() and g_trainer.net_secondary.is_converged():
            print("*** Both models are converged! ***")


            if  window_size_cnt >= args.window_size and not np.isnan(avg_loss_diff) and avg_loss_diff < LOSS_DIFF_THRESHOLD:
                print(f"Secondary model is tolarably good: {avg_loss_diff}, let's switch model")
                g_trainer.switch_model()
                window_size_cnt = 0
        
       
        # [Experiment #1] Static Freeze global model
        # if (e+1) % 20 == 0:
        #     mock_gradually_freezing_degree += 1
        #     for idx, l in enumerate(net_glob.layers):
        #         print(l)
        #         if (idx+1) <= mock_gradually_freezing_degree:
        #             l.requires_grad_(False)
        #     torchinfo.summary(net_glob, (1,3,32,32), device=args.device)
        



    end_time = time.time()
    print(f'Best model, iter: {best_epoch}, acc: {best_acc}')
    print(f'Total training time: {datetime.timedelta(seconds= end_time - start_time)}')
    accuracy = final_results.acc_test.to_numpy()


    print(g_trainer.net.loss_train)
    print(g_trainer.net.loss_test)
    print(g_trainer.net.loss_train_delta)
    print(g_trainer.net.loss_test_delta)
    print(g_trainer.net.acc)
    print(g_trainer.net_secondary.acc)

    myplotter.setup_plot("Global Metrics of FL w/ LeNet-5 Model on CIFAR 10 Dataset", "Loss", 1)
    myplotter.plot_data(g_trainer.net.loss_test, "Primary Test Loss")
    myplotter.plot_data(g_trainer.net_secondary.loss_test, "Secondary Test Loss")
    myplotter.plot_data(g_trainer.net.loss_train, "Primary Train Loss")

    myplotter.legend()
    myplotter.save_figure(base_dir, "global_model_metrics")
    # myplotter.plot_data(g_trainer.loss_train_delta, "Train Loss Delta")
    # myplotter.plot_data(g_trainer.loss_test_delta, "Test Loss Delta")
    
    myplotter.setup_plot("Global Metrics of FL w/ LeNet-5 Model on CIFAR 10 Dataset", "Accuracy", 2)
    myplotter.plot_data(g_trainer.net.acc, "Global Primary Model")
    myplotter.plot_data(g_trainer.net_secondary.acc, "Global Secondary Model")
    myplotter.legend()
    myplotter.save_figure(base_dir, "global_model_accuracy")

    
    myplotter.show()


    print(np.array2string(accuracy, separator=', '))
    