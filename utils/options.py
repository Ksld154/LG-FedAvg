#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--local_ep_pretrain', type=int, default=0, help="the number of pretrain local ep")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--print_freq', type=int, default=100, help="print loss frequency during training")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--load_fed', type=str, default='', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='/', help='define fed results save folder')
    parser.add_argument('--start_saving', type=int, default=0, help='when to start saving models')
    
    ### my arguments
    parser.add_argument('--window_size', type=int, default=5, help='Window size for moving avg. loss')
    parser.add_argument('--gradually_freezing', action='store_true', help='Enable Gradually Freezing')
    
    parser.add_argument('--switch_model', dest='switch_model', action='store_true', help='Enable Switch model in Gradually Freezing')
    parser.add_argument('--no-switch_model', dest='switch_model', action='store_false', help='Disable Switch model in Gradually Freezing')
    parser.set_defaults(switch_model=True)

    parser.add_argument('--brute_force', dest='brute_force', action='store_true', help='Brute force list every possiable freezing degree')
    parser.add_argument('--no-brute_force', dest='brute_force', action='store_false', help='Disable brute force list every possiable freezing degree')
    parser.set_defaults(brute_force=False)


    parser.add_argument('--static_freeze', dest='static_freeze', action='store_true', help='Static Freezing')
    parser.add_argument('--no-static_freeze', dest='static_freeze', action='store_false', help='Disable Static Freezing')
    parser.set_defaults(static_freeze=False)
    
    parser.add_argument('--load_pretrained', type=str, default='', help='Load pretrained  model path')
    parser.add_argument('--static_freeze_candidates', type=int, default=5, help='Static Freeze candidates numbers')
    parser.add_argument('--loss_diff_ratio', type=float, default=-1, help="loss_diff_ratio to switch model")

    parser.add_argument('--optimistic_train', dest='optimistic_train', action='store_true', help='Optimistic train and switch between models (Our method)')
    parser.add_argument('--no-optimistic_train', dest='optimistic_train', action='store_false', help='Do not optimistic train and switch between models (Baseline)')
    parser.set_defaults(optimistic_train=True)
    
    parser.add_argument('--pre_trained_rounds', type=int, default=15, help='Pre-trained rounds')

    parser.add_argument('--converged_threshold', type=float, default=0.05, help="loss_diff_ratio to switch model")

    args = parser.parse_args()
    return args
