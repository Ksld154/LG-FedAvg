#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
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
from models.trainer import GlobalTrainer, LocalTrainer
from utils.tools import moving_average
import utils.myplotter as myplotter

import pdb

def decide_further_freeze():
    pass

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f'GPU ID: {args.device}')

    base_dir = f'./save/{args.dataset}/{args.model}_iid{args.iid}_num{args.num_users}_C{args.frac}_le{args.local_ep}/shard{args.shard_per_user}/{args.results_save}/'
    if not os.path.exists(os.path.join(base_dir, 'fed')):
        os.makedirs(os.path.join(base_dir, 'fed'), exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    dict_save_path = os.path.join(base_dir, 'dict_users.pkl')
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)

    # build model
    net_glob = get_model(args)
    net_glob.train()
    g_trainer = GlobalTrainer(args= args, net=net_glob)

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

    current_time = datetime.datetime.now()
    start_time = time.time()
    print(f'Training Start: {current_time}')


    for e in range(args.epochs):
        g_trainer.weights = None
        lr *= args.lr_decay

        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)  # number of workers per round
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f'Round {(e+1):3d}, lr: {lr:.6f}, {idxs_users}')

        for idx in idxs_users:
            local_trainer = LocalTrainer(args=args, dataset=dataset_train, idxs=dict_users_train[idx])

            if switch_model_flag:
                pass

            # Train local primary model
            net_local = copy.deepcopy(g_trainer.net)
            w_local, loss = local_trainer.train(net=net_local.to(args.device), lr=lr)
            loss_locals.append(copy.deepcopy(loss))
            
            # Train local secondary model
            net_local_secondary = copy.deepcopy(net_glob)
            local_trainer.further_freeze(net=net_local_secondary, freeze_degree=1)
            w_local_secondary, loss_secondary = local_trainer.train(net=net_local_secondary.to(args.device), lr=lr)
            
            # increament global weights (aggregation)
            if g_trainer.weights is None:
                g_trainer.weights = copy.deepcopy(w_local)
            else: 
                for k in g_trainer.weights.keys():
                    g_trainer.weights[k] += w_local[k]

        # update global weights (aggregation), then update global model
        for k in g_trainer.weights.keys():
            g_trainer.weights[k] = torch.div(g_trainer.weights[k], m)
        g_trainer.net.load_state_dict(g_trainer.weights)


        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)

        if g_trainer.loss_train:
           loss_diff = loss_avg -  g_trainer.loss_train[-1]
           g_trainer.loss_train_delta.append(loss_diff)
        g_trainer.loss_train.append(loss_avg)


        # test/validate global model
        acc_test = loss_test = None
        if (e+1) % args.test_freq == 0:
            g_trainer.net.eval()
            acc_test, loss_test = test_img(g_trainer.net, dataset_test, args)
            print(f'Round {(e+1):3d}, Average loss {loss_avg:.3f}, Test loss {loss_test:.3f}, Test accuracy: {acc_test:.2f}')

            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(g_trainer.net)
                best_acc = acc_test
                best_epoch = e

            results.append(np.array([e, loss_avg, loss_test, acc_test, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
            final_results.to_csv(results_save_path, index=False)

            if g_trainer.loss_test:
                loss_diff = loss_test - g_trainer.loss_test[-1]
                g_trainer.loss_test_delta.append(loss_diff)
            g_trainer.loss_test.append(loss_test)


        if (e+1) % 50 == 0:
            best_save_path = os.path.join(base_dir, f'fed/best_{(e + 1)}.pt')
            model_save_path = os.path.join(base_dir, f'fed/model_{(e + 1)}.pt')
            torch.save(net_best.state_dict(), best_save_path)
            torch.save(g_trainer.net.state_dict(), model_save_path)
        
        if (e+1) % 10 == 0:
            current_time = datetime.datetime.now()
            now = time.time()
            print(f'Epoch {e+1}, Current Time: {current_time}')
            print(f'Elapsed Time: {datetime.timedelta(seconds= now - start_time)}')

        # [Experiment \#2] Gradually freezing(decision at global side, freezing at local side)
        avg_loss = moving_average(g_trainer.loss_test, 15)
        if not np.isnan(avg_loss) and avg_loss < 0.1:
            switch_model_flag = True
            # g_trainer.loss_test.clear()
        
        
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

    print(g_trainer.loss_train)
    print(g_trainer.loss_test)
    print(g_trainer.loss_train_delta)
    print(g_trainer.loss_test_delta)

    myplotter.setup_plot("Global Metrics of FL w/ LeNet-5 Model on CIFAR 10 Dataset", "Loss")
    myplotter.plot_data(g_trainer.loss_train, "Train Loss")
    myplotter.plot_data(g_trainer.loss_test, "Test Loss")
    myplotter.plot_data(g_trainer.loss_train_delta, "Train Loss Delta")
    myplotter.plot_data(g_trainer.loss_test_delta, "Test Loss Delta")
    myplotter.show()


    accuracy = final_results.acc_test.to_numpy()
    print(np.array2string(accuracy, separator=', '))
    