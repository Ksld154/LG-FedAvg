#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from multiprocessing.spawn import import_main_path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import pdb

import time 
import datetime
from constants import *

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

        self.trainable_params = None
        self.local_train_time = datetime.timedelta(seconds=0)
        self.upload_time = datetime.timedelta(seconds=0)
        self.download_time = datetime.timedelta(seconds=0)

    def train(self, net, idx=-1, lr=0.1):
        self.local_train_time = datetime.timedelta(seconds=0)
        train_start = time.time()

        # train and update
        net.train()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.5)
        # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        train_end = time.time()
        self.local_train_time = datetime.timedelta(seconds= train_end-train_start)
        
        total_params = sum(p.numel() for p in net.parameters())
        frozen_params = sum(p.numel() for p in net.parameters() if not p.requires_grad)
        self.trainable_params = total_params - frozen_params

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def calc_train_and_transmission_time(self, active_workers=1):
        train_time = self.local_train_time

        payload_size = self.trainable_params * 4 * 8 # (in bits)
        # print(payload_size)

        upload_bandwidth = random.uniform(MIN_UPLOAD_BANDWIDTH, MAX_UPLOAD_BANDWIDTH) * (10**6) / active_workers
        upload_time = payload_size / upload_bandwidth
        # print(upload_bandwidth)
        # print(upload_time)

        self.upload_time = datetime.timedelta(seconds=upload_time)
        # print(f'Train Time: {train_time}')
        # print(f'Upload Time: {self.upload_time}')

        download_bandwidth = random.uniform(MIN_DOWNLOAD_BANDWIDTH, MAX_DOWNLOAD_BANDWIDTH) * (10**6) / active_workers
        download_time = payload_size / download_bandwidth
        self.download_time = datetime.timedelta(seconds=download_time)
        # print(f'Download Time: {self.download_time}')

class LocalUpdateMTL(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, lr=0.1, omega=None, W_glob=None, idx=None, w_glob_keys=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)

                W = W_glob.clone()

                W_local = [net.state_dict(keep_vars=True)[key].flatten() for key in w_glob_keys]
                W_local = torch.cat(W_local)
                W[:, idx] = W_local

                loss_regularizer = 0
                loss_regularizer += W.norm() ** 2

                k = 4000
                for i in range(W.shape[0] // k):
                    x = W[i * k:(i+1) * k, :]
                    loss_regularizer += x.mm(omega).mm(x.T).trace()
                f = (int)(math.log10(W.shape[0])+1) + 1
                loss_regularizer *= 10 ** (-f)

                loss = loss + loss_regularizer
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


