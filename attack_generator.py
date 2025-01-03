import numpy as np
from models import *
from torch.autograd import Variable
from main import calculate_loss

def cwloss(output, target,confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

# Geometry-aware projected gradient descent (GA-PGD)
def GA_PGD(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    Kappa = torch.zeros(len(data))
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_output = model(data)
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        _, output = model(x_adv)
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, Kappa

def eval_clean(model, args, agg, test_loader, epoch):
    dict_name = 'test_clean'
    model.eval()
    test_loss = 0
    correct = 0
    num_data = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            num_data += target.shape[0]
            _, output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)

    num_batches = len(test_loader)
    agg[f"{dict_name}_loss"].append(test_loss)
    agg[f"{dict_name}_acc"].append(test_accuracy)

    agg[f"{dict_name}_aux_loss"][epoch] /= num_batches
    agg[f"{dict_name}_ce_loss"][epoch] /= num_batches

    print(
        "Epoch {}/{}: {} Acc: {} ({}/{})".format(
            epoch,
            args.epochs,
            dict_name,
            agg[f"{dict_name}_acc"][epoch],
            agg[f"{dict_name}_running_corrects"][epoch],
            num_data,
        ),
        end=" ",
    )
    print(
        "| {} Total Loss: {} ({}+{})".format(
            dict_name,
            agg[f"{dict_name}_loss"][epoch],
            agg[f"{dict_name}_aux_loss"][epoch],
            agg[f"{dict_name}_ce_loss"][epoch],
        )
    )

    return test_loss, test_accuracy

def eval_robust(model, args, agg, test_loader, epoch, loss_fn, category, random):
    dict_name = 'test_adv'
    model.eval()
    test_loss = 0
    correct = 0
    num_data = 0
    with torch.enable_grad():
        for data, target in test_loader:
            num_data += target.shape[0]
            data, target = data.cuda(), target.cuda()
            x_adv, _ = GA_PGD(model, data,target, args.epsilon, args.step_size, args.num_steps ,loss_fn,category,rand_init=random)
            _, output = model(x_adv)
            test_loss += calculate_loss(args, model, agg, epoch, data, target, x_adv, phase=dict_name).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)

    num_batches = len(test_loader)
    agg[f"{dict_name}_loss"].append(test_loss)
    agg[f"{dict_name}_acc"].append(test_accuracy)

    agg[f"{dict_name}_aux_loss"][epoch] /= num_batches
    agg[f"{dict_name}_ce_loss"][epoch] /= num_batches

    print(
        "Epoch {}/{}: {} Acc: {} ({}/{})".format(
            epoch,
            args.epochs,
            dict_name,
            agg[f"{dict_name}_acc"][epoch],
            agg[f"{dict_name}_running_corrects"][epoch],
            num_data,
        ),
        end=" ",
    )
    print(
        "| {} Total Loss: {} ({}+{})".format(
            dict_name,
            agg[f"{dict_name}_loss"][epoch],
            agg[f"{dict_name}_aux_loss"][epoch],
            agg[f"{dict_name}_ce_loss"][epoch],
        )
    )
    
    return test_loss, test_accuracy

