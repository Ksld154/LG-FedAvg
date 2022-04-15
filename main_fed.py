#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch
import datetime
import time
import torchinfo

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalTrainer, LocalUpdate
from models.test import test_img
import os

from models.trainer import GlobalTrainer, LocalTrainer

import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f'GPU ID: {args.gpu}')

    base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/{}/'.format(
        args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.results_save)
    if not os.path.exists(os.path.join(base_dir, 'fed')):
        os.makedirs(os.path.join(base_dir, 'fed'), exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    dict_save_path = os.path.join(base_dir, 'dict_users.pkl')
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)

    # build model
    net_glob = get_model(args)
    net_glob.train()

    # training
    results_save_path = os.path.join(base_dir, 'fed/results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []
    # tt = torch.Tensor([1,3,32,32])

    current_time = datetime.datetime.now()
    start_time = time.time()
    print(f'Training Start: {current_time}')
    
    # global_trainer = GlobalTrainer(net_glob)

    for e in range(args.epochs):
        w_glob = None
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)  # number of workers per round
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Round {}, lr: {:.6f}, {}".format(e+1, lr, idxs_users))

        for idx in idxs_users:
            # local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            local = LocalTrainer(args=args, dataset=dataset_train, idxs=dict_users_train[idx])

            net_local = copy.deepcopy(net_glob)

            w_local, loss = local.train(net=net_local.to(args.device), lr=lr)
            loss_locals.append(copy.deepcopy(loss))
            
            # freeze every 10 epochs
            # if (e+1) % 2 == 0:
            #     net_local = local.further_freeze(net=net_local.to(args.device), freeze_degree=1)

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
            
            # if (e+1) % 6 == 0:
            #     torchinfo.summary(net_local.to(args.device), (1,3,32,32), device=args.device)


        lr *= args.lr_decay

        # update global weights
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (e + 1) % args.test_freq == 0:
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                e+1, loss_avg, loss_test, acc_test))


            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = e

            results.append(np.array([e, loss_avg, loss_test, acc_test, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
            final_results.to_csv(results_save_path, index=False)

        if (e + 1) % 50 == 0:
            best_save_path = os.path.join(base_dir, 'fed/best_{}.pt'.format(e + 1))
            model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(e + 1))
            torch.save(net_best.state_dict(), best_save_path)
            torch.save(net_glob.state_dict(), model_save_path)
        
        if (e+1) % 10 == 0:
            current_time = datetime.datetime.now()
            now = time.time()
            print(f'Epoch {e+1}, Current Time: {current_time}')
            print(f'Elapsed Time: {datetime.timedelta(seconds= now - start_time)}')


        # freeze global model??
        if (e+1) % 20 == 0:
            for idx, l in enumerate(net_glob.layers):
                print(l)
                if (idx+1) <= 0:
                    l.requires_grad_(False)

            torchinfo.summary(net_glob, (1,3,32,32), device=args.device)
        


    end_time = time.time()
    print(f'Best model, iter: {best_epoch}, acc: {best_acc}')
    print(f'Total training time: {datetime.timedelta(seconds= end_time - start_time)}')
