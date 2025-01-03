import os
import sys
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
from models import *
from GAIR import GAIR
import numpy as np
import attack_generator as attack
from pathlib import Path
from tempfile import mkdtemp
from collections import defaultdict
from tqdm import trange, tqdm
import json
from utils import loading_utils




from utils import constants
from utils.plot_utils import custom_plot_loss, plot_conf, make_dir

from utils.log import (
    Logger,
    Timer,
    save_model,
    save_vars,
    DefaultList,
    save_model_checkpoint,
)


parser = argparse.ArgumentParser(description='GAIRAT: Geometry-aware instance-dependent adversarial training')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')

## PGD settings
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step-size', type=float, default=0.007, help='step size')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="WRN",help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn,cifar100,mnist")
parser.add_argument('--random',type=bool,default=True,help="whether to initiat adversarial sample with random noise")
parser.add_argument('--depth',type=int,default=32,help='WRN depth')
parser.add_argument('--width-factor',type=int,default=10,help='WRN width factor')
parser.add_argument('--drop-rate',type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--resume',type=str,default=None,help='whether to resume training')

parser.add_argument('--experiment',type=str, required=True,help='dir of output')

# parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'onedrop', 'multipledecay', 'cosine'])
# parser.add_argument('--lr-max', default=0.1, type=float)
# parser.add_argument('--lr-one-drop', default=0.01, type=float)
# parser.add_argument('--lr-drop-epoch', default=100, type=int)

parser.add_argument('--objective',  choices=['AT', 'GAIRAT', 'AFP', 'GAIRAT_AFP'], required=True)

## GAIRAT settings
parser.add_argument('--Lambda',type=str, default='-1.0', help='parameter for GAIR')
parser.add_argument('--Lambda_max',type=float, default=float('inf'), help='max Lambda')
parser.add_argument('--Lambda_schedule', default='fixed', choices=['linear', 'piecewise', 'fixed'])
parser.add_argument('--weight_assignment_function', default='Tanh', choices=['Discrete','Sigmoid','Tanh'])
parser.add_argument('--begin_epoch', type=int, default=60, help='when to use GAIR')

parser.add_argument('--alpha', type=float, default=0.01, help='The coefficient of the feature pairwise loss')
parser.add_argument("--warmup", type=int, default=0, help="number of epochs for warming up the model with CE, for joint supervision objectives, default: (0, no warming up) ")

parser.add_argument("--lr", type=float, default=0.001, help="learning rate, default: (0.001)")

parser.add_argument(
    "--scheduler",
    type=str,
    default="none",
    metavar="M",
    choices=["StepLR", "MultiStepLR"],
    help="optimizer (default: none)",
)
parser.add_argument(
    "--lr_gamma",
    type=float,
    default=1.0,
    help="SCHEDULING, Multiplicative factor of learning rate decay , default: (1.0, no scheduling) ",
)
parser.add_argument(
    "--lr_steps",
    metavar="N",
    type=int,
    nargs="+",
    default=[],
    help="List of epoch indices. Must be increasing/Period of learning rate decay",
)

parser.add_argument(
    "--optim",
    type=str,
    default="adam",
    metavar="M",
    choices=["sgd", "adam"],
    help="optimizer (default: adam) - the sgd option is with momentum of 0.9",
)

args = parser.parse_args()


# Save checkpoint0
def save_checkpoint(state, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# Get adversarially robust network
def train(epoch, agg, model, train_loader, optimizer, Lambda):
    
    lr = 0
    num_data = 0
    train_robust_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        loss = 0
        data, target = data.cuda(), target.cuda()
        
        # Get adversarial data and geometry value
        x_adv, Kappa = attack.GA_PGD(model, data, target, args.epsilon, args.step_size, args.num_steps, loss_fn="cent",category="Madry", rand_init=True)

        model.train()
        # lr = lr_schedule(epoch + 1)
        # optimizer.param_groups[0].update(lr=lr)
        optimizer.zero_grad()
        
        loss = calculate_loss(args, model, agg, epoch, data, target, x_adv, Lambda, Kappa)
        train_robust_loss += loss.item() * len(x_adv)
        
        loss.backward()
        optimizer.step()
        
        num_data += len(data)

    train_robust_loss = train_robust_loss / num_data

    num_batches = len(train_loader)
    agg["train_loss"].append(train_robust_loss)
    agg["train_acc"].append(agg["train_running_corrects"][epoch] / num_data)

    agg["train_aux_loss"][epoch] /= num_batches
    agg["train_ce_loss"][epoch] /= num_batches
    agg["train_pairwise_loss"][epoch] /= num_batches

    print(
        "Epoch {}/{}: Train Acc: {} ({}/{})".format(
            epoch,
            args.epochs,
            agg["train_acc"][epoch],
            agg["train_running_corrects"][epoch],
            num_data,
        ),
        end=" ",
    )
    print(
        "| Train Total Loss: {} ({}+{})".format(
            agg["train_loss"][epoch],
            agg["train_aux_loss"][epoch],
            agg["train_ce_loss"][epoch],
        )
    )

    return train_robust_loss, lr

# Calculate Loss
def calculate_loss(args, model, agg, epoch, data, target, x_adv, Lambda=None, Kappa=None, phase='train'):
    adv_feats, logit = model(x_adv)
    _, preds = torch.max(logit.data, dim=1)
    agg[f"{phase}_running_corrects"][epoch] += (preds == target).sum().item()

    if args.objective == 'AT' or (args.objective == 'GAIRAT' and phase[0:4] == 'test'):
        ce_loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)
        agg[f"{phase}_ce_loss"][epoch] += ce_loss.item()
        return ce_loss
    elif args.objective == 'GAIRAT':
        # Based on "https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training"
        if (epoch + 1) >= args.begin_epoch:
            Kappa = Kappa.cuda()
            loss = nn.CrossEntropyLoss(reduce=False)(logit, target)
            # Calculate weight assignment according to geometry value
            normalized_reweight = GAIR(args.num_steps, Kappa, Lambda, args.weight_assignment_function)
            loss = loss.mul(normalized_reweight).mean()
            loss.mul(normalized_reweight).mean()
            agg[f"{phase}_ce_loss"][epoch] += loss.item()
            return loss
        else:
            loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)
            agg[f"{phase}_ce_loss"][epoch] += loss.item()
            return loss
    
    elif args.objective == 'AFP' or (args.objective == 'GAIRAT_AFP' and phase[0:4] == 'test'):
        ce_loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)
        agg[f"{phase}_ce_loss"][epoch] += ce_loss.item()
        if epoch < args.warmup:
            return ce_loss
        else:
            mse = nn.MSELoss(reduction="none")
            cln_feats, _  = model(data)
            pairwise_loss = 0.5 * mse(adv_feats, cln_feats).mean()
            agg[f"{phase}_aux_loss"][epoch] += pairwise_loss.item()
            return ce_loss + args.alpha * pairwise_loss
        
    elif args.objective == 'GAIRAT_AFP':
        mse = nn.MSELoss(reduction="none")
        cln_feats, _  = model(data)

        if (epoch + 1) >= args.begin_epoch:
            Kappa = Kappa.cuda()
            pairwise_loss = 0.5 * mse(adv_feats, cln_feats).mean(dim=1)
            ce_loss = nn.CrossEntropyLoss(reduce=False)(logit, target)
            # print("ce loss", ce_loss.shape)
            # print("pairwise loss", pairwise_loss.shape)
            # Calculate weight assignment according to geometry value
            normalized_reweight = GAIR(args.num_steps, Kappa, Lambda, args.weight_assignment_function)
            if epoch < args.warmup:
                ce_loss = ce_loss.mul(normalized_reweight).mean()
                agg[f"{phase}_ce_loss"][epoch] += ce_loss.item()
                return ce_loss
            else:
                pairwise_loss =  pairwise_loss.mul(normalized_reweight).mean()
                ce_loss = ce_loss.mul(normalized_reweight).mean()
                agg[f"{phase}_aux_loss"][epoch] += pairwise_loss.item()
                agg[f"{phase}_ce_loss"][epoch] += ce_loss.item()
                return ce_loss + args.alpha * pairwise_loss
        else:
            ce_loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)
            agg[f"{phase}_ce_loss"][epoch] += ce_loss.item()
            if epoch < args.warmup:
                return ce_loss
            else:
                mse = nn.MSELoss(reduction="none")
                cln_feats, _  = model(data)
                pairwise_loss = 0.5 * mse(adv_feats, cln_feats).mean()
                agg[f"{phase}_aux_loss"][epoch] += pairwise_loss.item()
                return ce_loss + args.alpha * pairwise_loss

# Adjust lambda for weight assignment using epoch
def adjust_Lambda(epoch):
    Lam = float(args.Lambda)
    if args.epochs >= 110:
        # Train Wide-ResNet
        Lambda = args.Lambda_max
        if args.Lambda_schedule == 'linear':
            if epoch >= 60:
                Lambda = args.Lambda_max - (epoch/args.epochs) * (args.Lambda_max - Lam)
        elif args.Lambda_schedule == 'piecewise':
            if epoch >= 60:
                Lambda = Lam
            elif epoch >= 90:
                Lambda = Lam-1.0
            elif epoch >= 110:
                Lambda = Lam-1.5
        elif args.Lambda_schedule == 'fixed':
            if epoch >= 60:
                Lambda = Lam
    else:
        # Train ResNet
        Lambda = args.Lambda_max
        if args.Lambda_schedule == 'linear':
            if epoch >= 30:
                Lambda = args.Lambda_max - (epoch/args.epochs) * (args.Lambda_max - Lam)
        elif args.Lambda_schedule == 'piecewise':
            if epoch >= 30:
                Lambda = Lam
            elif epoch >= 60:
                Lambda = Lam-2.0
        elif args.Lambda_schedule == 'fixed':
            if epoch >= 30:
                Lambda = Lam
    return Lambda

if __name__ == '__main__':
    experiment_dir = Path(f"{constants.Constants.OUTPUT_DIR}/" + args.experiment)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = mkdtemp(dir=str(experiment_dir))

    sys.stdout = Logger("{}/run.log".format(runPath))
    command_line_args = sys.argv
    command = " ".join(command_line_args)

    # Training settings
    seed = args.seed
    momentum = args.momentum
    weight_decay = args.weight_decay
    depth = args.depth
    width_factor = args.width_factor
    drop_rate = args.drop_rate
    resume = args.resume

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Models and optimizer
    if args.net == "smallcnn":
        model = SmallCNN().cuda()
        net = "smallcnn"
    if args.net == "ResNet18_Mart":
        model = ResNet18_Mart().cuda()
        net = "ResNet18_Mart"
    if args.net == "preactresnet18":
        model = PreActResNet18().cuda()
        net = "preactresnet18"
    if args.net == "WRN":
        model = Wide_ResNet_Madry(depth=depth, num_classes=10, widen_factor=width_factor, dropRate=drop_rate).cuda()
        net = "WRN{}-{}-dropout{}".format(depth,width_factor,drop_rate)
    if args.net == "Madry":
        model = Madry().cuda()
        net = "Madry"

    model = torch.nn.DataParallel(model)
    if args.optim == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            amsgrad=args.amsgrad,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        # , nesterov=True
    )
    print("model optimizer: ", optimizer.state_dict())
    # optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=momentum, weight_decay=weight_decay)

    # Learning schedules
    opt_dict = {"optimizer": optimizer}
    schedulers = loading_utils.get_schedulers(args, opt_dict)


    # # Store path
    # if not os.path.exists(runPath):
    #     os.makedirs(runPath)


    with open("{}/args.json".format(runPath), "w") as fp:
        json.dump(args.__dict__, fp)
    torch.save(args, "{}/args.rar".format(runPath))


    print("Expt:", runPath)
    print(f"The command that ran this script: {command}")
    print("args: \n", args)

    models_dir = f"{runPath}/models"
    make_dir(models_dir)

    # Setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    if args.dataset == "svhn":
        trainset = torchvision.datasets.SVHN(root='./data/SVHN', split='train', download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testset = torchvision.datasets.SVHN(root='./data/SVHN', split='test', download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    if args.dataset == "mnist":
        trainset = torchvision.datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1,pin_memory=True)
        testset = torchvision.datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1,pin_memory=True)

    print(len(train_loader))
    # Resume 
    title = 'GAIRAT'
    best_acc = 0
    start_epoch = 0
    if resume:
        # Resume directly point to checkpoint.pth.tar
        print ('==> GAIRAT Resuming from checkpoint ..')
        print(resume)
        assert os.path.isfile(resume)
        runPath = os.path.dirname(resume)
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['test_pgd20_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger_test = Logger(os.path.join(runPath, 'log_results.txt'), title=title, resume=True)
    else:
        print('==> GAIRAT')
        # logger_test = Logger(os.path.join(runPath, 'log_results.txt'), title=title)
        # logger_test.set_names(['Epoch', 'Natural Test Acc', 'PGD20 Acc'])
        ## Training get started
        test_nat_acc = 0
        test_pgd20_acc = 0

    with Timer("Neural-Net-Pr") as t:
        agg = defaultdict(DefaultList)
        agg["optimizer"] = optimizer
        best_eval_loss = None
        best_eval_acc = None
        best_eval_acc_adv = None

        for epoch in trange(start_epoch, args.epochs):
            
            # Get lambda
            Lambda = adjust_Lambda(epoch + 1)
            
            # Adversarial training
            train_robust_loss, lr = train(epoch, agg, model, train_loader, optimizer, Lambda)

            # Evalutions similar to DAT.
            _, test_nat_acc = attack.eval_clean(model, args, agg, test_loader, epoch)
            _, test_pgd20_acc = attack.eval_robust(model, args, agg, test_loader, epoch, loss_fn="cent", category="Madry", random=True)

            for scheduler in schedulers:
                scheduler.step()

            print(
                'Epoch: [%d | %d] | Learning Rate: %f | Natural Test Acc %.2f | PGD20 Test Acc %.2f |\n' % (
                epoch,
                args.epochs,
                lr,
                test_nat_acc,
                test_pgd20_acc)
                )
            
            if (best_eval_acc is None or agg["test_clean_acc"][-1] >= best_eval_acc):
                save_model(model, models_dir + f"/model_best_acc.rar")
            if best_eval_acc_adv is None or agg["test_adv_acc"][-1] >= best_eval_acc_adv:
                save_model(model, models_dir + f"/model_best_adv_acc.rar")
            save_model(model, models_dir + f"/model_last.rar")
            save_model(optimizer, models_dir + f"/optim_last.rar")

            for _indx, _scheduler in enumerate(schedulers):
                save_model(_scheduler, models_dir + f"/scheduler{_indx}_last.rar")

            # logger_test.append([epoch + 1, test_nat_acc, test_pgd20_acc])

            custom_plot_loss(
                    agg,
                    ["train_loss", "test_clean_loss"],
                    "losses",
                    "loss",
                    runPath,
                )
            custom_plot_loss(
                agg,
                ["train_acc", f"test_clean_acc"],
                "acc",
                "acc",
                runPath,
                False,
            )
            custom_plot_loss(
                agg,
                ["train_aux_loss", "train_ce_loss"],
                "train all losses",
                "train all losses",
                runPath,
            )
            # custom_plot_loss(
            #     agg,
            #     ["train_pairwise_loss", "train_ce_loss"],
            #     "train pairwise\ce losses",
            #     "train pairwise\ce losses",
            #     runPath,
            # )
            custom_plot_loss(
                agg,
                ["test_clean_aux_loss", "test_clean_ce_loss"],
                "test all losses",
                "test all losses",
                runPath,
            )
            custom_plot_loss(
                agg,
                ["train_acc", "test_adv_acc"],
                "acc",
                "acc_adv",
                runPath,
                False,
            )
            custom_plot_loss(
                agg,
                ["train_loss", "test_adv_loss"],
                "losses_adv",
                "losses_adv",
                runPath,
            )
            custom_plot_loss(
                agg,
                ["test_adv_aux_loss", "test_adv_ce_loss"],
                "test all losses _adv",
                "test all losses _adv",
                runPath,
            )
            
            # Save the best checkpoint
            if test_pgd20_acc > best_acc:
                best_acc = test_pgd20_acc
                save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'test_nat_acc': test_nat_acc, 
                        'test_pgd20_acc': test_pgd20_acc,
                        'optimizer' : optimizer.state_dict(),
                    }, checkpoint=runPath, filename='bestpoint.pth.tar')

            # Save the last checkpoint
            save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'test_nat_acc': test_nat_acc, 
                        'test_pgd20_acc': test_pgd20_acc,
                        'optimizer' : optimizer.state_dict(),
                    }, checkpoint=runPath)
            
        # logger_test.close()