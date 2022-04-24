import copy
from multiprocessing.spawn import import_main_path
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from tqdm import tqdm
import math
import pdb
import torchinfo
from constants import LOSS_DELTA_CONVERGED_THRESHOLD

from utils.tools import moving_average
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

class GlobalTrainer(object):
    def __init__(self, args, net, net_secondary):
        self.args = args
        self.net = MyModel(model=net, args=args, freeze_degree=0)
        self.net_secondary = MyModel(model=net_secondary, args=args, freeze_degree=1)
        self.weights = None
        self.weights_secondary = None
        
        self.models_loss_test_diff = []

        self.brute_force_nets = [None]*5
        # self.brute_force_weights = []
        

    def switch_model(self):
        self.net.model = copy.deepcopy(self.net_secondary.model)
        if self.net.freeze_degree < 4:
            self.net.freeze_degree += 1
            self.net.further_freeze()

        # self.net_secondary = MyModel(model=copy.deepcopy(self.net_secondary.model), freeze_degree=self.net_secondary.freeze_degree+1, args=self.args)        
        if self.net_secondary.freeze_degree < 4:
            self.net_secondary.freeze_degree += 1

    def brute_force_search_models(self, old_primary_weights ,epoch):
        for d in range(4):
            print(f'Round {(epoch+1):3d}, building brute-force search model w/ degree: {d}')
            # old_weights = self.brute_force_nets[d].model.state_dict()
            new_weights = copy.deepcopy(self.weights)   
            
            if (epoch+1) > WARM_UP_ROUNDS:
                for idx, k in enumerate(new_weights.keys()):
                    # times 2 for weights and bias in single layer (LeNet-5)
                    if idx < self.brute_force_nets[d].freeze_degree * 2:  # use old weights
                        # print(k)
                        new_weights[k] = copy.deepcopy(old_primary_weights[k])
            self.brute_force_nets[d].model.load_state_dict(new_weights)
    
    # copy from current primary model, for Gradually Freezing
    def generate_secondary_model_method_1(self, old_primary_weights, epoch):
        new_weights = copy.deepcopy(self.weights) # Weights after aggregation
    
        if (epoch+1) > WARM_UP_ROUNDS:
            for idx, k in enumerate(new_weights.keys()):
                # times 2 for weights and bias in single layer (LeNet-5)
                if idx < self.net_secondary.freeze_degree * 2:  
                    # use old weights
                    new_weights[k] = copy.deepcopy(old_primary_weights[k])
                    print(k)
                    # print(old_weights['conv1.bias'])
                    # print(g_trainer.weights[k])
                    # print(g_trainer.weights_secondary[k])
            # print(old_weights['conv1.bias'])
            # print(g_trainer.weights['conv1.bias'])
            # print(g_trainer.weights_secondary['conv1.bias'])
        self.net_secondary.model.load_state_dict(new_weights)
    

    # independently maintain a secondary model, for FreezeOut
    def generate_secondary_model_method_2(self, epoch):
        new_weights = copy.deepcopy(self.weights) # Weights after aggregation  
        old_secondary_weights = self.net_secondary.model.state_dict() 
        
        if (epoch+1) > WARM_UP_ROUNDS:
            for idx, k in enumerate(new_weights.keys()):
                # times 2 for weights and bias in single layer (LeNet-5)
                if idx < self.net_secondary.freeze_degree * 2:  # use old weights
                    new_weights[k] = copy.deepcopy(old_secondary_weights[k])
                    print(k)
        self.net_secondary.model.load_state_dict(new_weights)
    

    def test_model(self):
        pass

class LocalTrainer(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

        self.freeze_degree = 0

        self.net_primary = MyModel(model=None, args=args, freeze_degree=0)
        # self.net_secondary = MyModel(model=None, args=args)
        self.model_loss_diff = []


    def train(self, net, lr=0.1):
        # train and update
        net.train()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.5)
        # torchinfo.summary(net, (1, 3, 32, 32))

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
                # print(images.shape)
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)





class MyModel(object):
    def __init__(self, model, args, freeze_degree) -> None:
        self.model = model
        self.args = args
        self.freeze_degree = freeze_degree
        
        self.loss_train = []
        self.loss_train_delta = []
        self.acc = []

        self.loss_test = []
        self.loss_test_delta = []
    
    def update_loss_train_delta(self, loss):
        if self.loss_train:
            self.loss_train_delta.append(loss - self.loss_train[-1])

    def update_loss_test_delta(self, loss):
        if self.loss_test:
            self.loss_test_delta.append(loss - self.loss_test[-1])

    def is_converged(self):
        avg_train_loss_delta = moving_average(self.loss_train_delta, self.args.window_size)
        if not np.isnan(avg_train_loss_delta) and avg_train_loss_delta < LOSS_DELTA_CONVERGED_THRESHOLD:
            return True
        return False
    

    def further_freeze(self):
        for idx, l in enumerate(self.model.layers):
            if (idx+1) <= self.freeze_degree:
                l.requires_grad_(False)
        # torchinfo.summary(net, (1, 3, 32, 32), device=self.args.device)
        # return net