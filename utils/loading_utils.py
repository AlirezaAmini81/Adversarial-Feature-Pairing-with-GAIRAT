import sys
import os
import argparse
import models

# relative import hacks (sorry)
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # for bash user
os.chdir(parentdir)  # for pycharm user
from utils import constants
from utils.custom_data import get_mnist_data, get_cifar10_data, get_gcn_zca_cifar10_data
import utils.loss_functions as loss_functions
import torch
import numpy as np
import random
import torch.nn as nn




def get_params_to_update(model_ft):
    print("Params to learn:")
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)

    return params_to_update


from typing import Dict


def get_schedulers(args, optimizers: Dict):
    schedulers = []
    if args.scheduler == "none":
        print(f"No learning rate scheduling... {args.lr_gamma}, {args.lr_steps}")
    elif args.scheduler == "StepLR":
        from torch.optim.lr_scheduler import StepLR

        milestones = args.lr_steps
        assert len(milestones) == 1, "Use one milestone for StepLR scheduler..."
        for opt_k in optimizers.keys():
            schedulers.append(
                StepLR(
                    optimizers[opt_k],
                    step_size=milestones[0],
                    gamma=args.lr_gamma,
                    verbose=True,
                )
            )
            print(f"scheduler for {opt_k}: ", schedulers[-1].state_dict())

    elif args.scheduler == "MultiStepLR":
        from torch.optim.lr_scheduler import MultiStepLR

        milestones = args.lr_steps
        assert len(milestones) >= 1, "Set milestones for MultiStepLR scheduler..."
        for opt_k in optimizers.keys():
            schedulers.append(
                MultiStepLR(
                    optimizers[opt_k],
                    milestones=milestones,
                    gamma=args.lr_gamma,
                    verbose=True,
                )
            )
            print(f"scheduler for {opt_k}: ", schedulers[-1].state_dict())
    else:
        print(f"scheduler {args.scheduler} not supported!")
        exit()

    return schedulers


def get_data(args, shuffle=True):
    # Dataloader:
    if args.dataset == "mnist":
        args.normalize = not args.adversarial_training

        ### DELETE LATER
        if args.temp_no_normalize:
            args.normalize = False

        if not "validation" in list(vars(args).keys()):
            args.validation = False

        print(args.normalize)
        full_loaders, input_size, num_classes, _ = get_mnist_data(
            data_loader_seed=args.seed,
            uniform_sampler=args.uniform_sampler,
            batch_size=args.batch_size,
            normalize=args.normalize,
            drop_last=False,
            validation=args.validation,
            shuffle=shuffle,
        )

    elif args.dataset == "cifar10":
        # delete later
        args.normalize = not args.adversarial_training

        ## TODO: set args.normalize accordinly
        ## TODO: set get_cifar10_data() to read from Constants.Cifar10_dir
        augment = "madry" if args.adversarial_training else "default"
        if args.no_augment:
            augment = "false"
        if not args.gcn_zca:
            resnet_transform = args.model == "RESNET18"
            full_loaders, input_size, num_classes, _ = get_cifar10_data(
                data_loader_seed=args.seed,
                uniform_sampler=args.uniform_sampler,
                batch_size=args.batch_size,
                adv_transform=args.adv_transform,
                resnet_transform=resnet_transform,
                drop_last=False,
                augment=augment,
                shuffle=shuffle,
            )
        else:
            # get_gcn_zca_cifar10_data
            full_loaders, input_size, num_classes, _ = get_gcn_zca_cifar10_data(
                data_loader_seed=args.seed,
                uniform_sampler=args.uniform_sampler,
                # augment=not args.no_cifar10_augment,
                batch_size=args.batch_size,
                # v2 = args.gcn_v2,
            )
    else:
        print("unsupported dataset")

    args.num_classes = num_classes
    return full_loaders, input_size, num_classes


def set_reproducability(seed):
    # random seed
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
