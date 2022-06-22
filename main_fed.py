#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import datetime
import pickle
import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img
import utils.myplotter
import utils.csv_exporter

import os
import pdb
import torchinfo

class Experiment():
    def plot_figure(self, all_results, output_dir):
        utils.myplotter.multiplot(
                all_data = all_results, 
                y_label="Accuracy",
                title="FL Static Freezing Accuracy", 
                figure_idx=1
            )

        utils.myplotter.legend()
        utils.myplotter.save_figure(output_dir, "FL_Static_Freezing_Accuracy.png")
        utils.myplotter.show()
    
    def output_csv(self, data, output_dir, fields):
        csv_file = os.path.join(output_dir, "result.csv")
        print(csv_file)
        utils.csv_exporter.export_csv(data=data, filepath=csv_file, fields=fields)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

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
    freeze_degree = 0 # This is for Static Freezing!!!

    for iter in range(args.epochs):
        w_glob = None
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = copy.deepcopy(net_glob)

            w_local, loss = local.train(net=net_local.to(args.device))
            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]

        lr *= args.lr_decay

        # update global weights
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % args.test_freq == 0:
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))


            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter

            results.append(np.array([iter, loss_avg, loss_test, acc_test, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
            final_results.to_csv(results_save_path, index=False)

        if (iter + 1) % 50 == 0:
            best_save_path = os.path.join(base_dir, 'fed/best_{}.pt'.format(iter + 1))
            model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(iter + 1))
            torch.save(net_best.state_dict(), best_save_path)
            torch.save(net_glob.state_dict(), model_save_path)

        # [Experiment #1] Static Freeze global model
        if args.static_freeze and (iter+1) % 20 == 0:
            for idx, l in enumerate(net_glob.layers):
                # print(l)
                if (idx+1) <= freeze_degree:
                    l.requires_grad_(False)

            torchinfo.summary(net_glob, (1,3,32,32), device=args.device)

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
    accuracy = final_results.acc_test.to_numpy()
    print(np.array2string(accuracy, separator=', '))
    print(f'Static Freeze Degree: {freeze_degree}')

    acc_list = list(accuracy)
    acc_list = [x / 100.0  for x in acc_list]
    all_results = []
    all_results.append(dict(name=f'Static Freeze: {freeze_degree} layers', acc=acc_list))


    script_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    result_dir = f'./save/{args.dataset}/{args.model}_iid{args.iid}_num{args.num_users}_C{args.frac}_le{args.local_ep}/shard{args.shard_per_user}/{args.results_save}/{script_time}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    exp1 = Experiment()
    exp1.output_csv(data=all_results, output_dir=result_dir, fields=['name', 'acc'])
    exp1.plot_figure(all_results=all_results, output_dir=result_dir)