import torch
import numpy as np
import torch.nn.parallel
from torch.autograd import Variable
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from utils.loss_function_utils import get_c_var


def validate_svm(model, train_loader, data_loader, is_test=False, lin_clf=svm.LinearSVC(), is_prob=False, is_cuda=False):
    model.eval()
    correct = 0
    train_out = None
    if not is_test:
        for n_iter, (data, target) in enumerate(train_loader):
            if is_cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            if train_out is None:
                train_out = model(data)[1].data
                target_out = target.data
            else:
                train_out = torch.cat((train_out, model(data)[1].data), 0)
                target_out = torch.cat((target_out, target.data))
        train_out = train_out.cpu().numpy()
        target_out = target_out.cpu().numpy()
        # index0 = target_out == 0
        # index1 = target_out == 1
        # plt.plot(train_out[index0, 0], train_out[index0, 1], 'b.')
        # plt.plot(train_out[index1, 0], train_out[index1, 1], 'r+')
        # plt.show()
        classifier = svm.LinearSVC()
        if n_iter % 1000 == 0:
            print('validation iter: ', n_iter)
        # classifier = svm.SVC(kernel='linear')
        print('training SVM')
        classifier.fit(train_out, target_out)

    pred_prob = None
    pred_labels = None
    target_labels = None
    all_out = None
    num_data = 0
    for data, target in data_loader:
        num_data += target.shape[0]
        if is_cuda:
            data= data.cuda()
        with torch.no_grad():
            data = Variable(data)
        output = model(data)[1].data.cpu().numpy()

        if all_out is None:
            all_out = output
        else:
            all_out = np.vstack((all_out, output))

        if is_prob:
            prob = lin_clf.decision_function(output)
            if pred_prob is None:
                pred_prob = prob[:, 1]
            else:
                pred_prob = np.hstack((pred_prob, prob[:, 1]))

        if not is_test:
            predict = classifier.predict(output)
        else:
            predict = lin_clf.predict(output)
        predict = torch.LongTensor(predict)
        correct += predict.eq(target).cpu().sum()
        if pred_labels is None:
            pred_labels = predict.numpy()
            target_labels = target.cpu().numpy()
        else:
            pred_labels = np.hstack((pred_labels, predict.numpy()))
            target_labels = np.hstack((target_labels, target.cpu().numpy()))

    pred_acc = 100. * float(correct) / num_data
    print('Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(num_data),pred_acc))

    if not is_test:
        return target_labels, pred_labels, pred_acc, classifier
    else:
        return target_labels, pred_labels, pred_acc, all_out


class epoch_stats:
    def __init__(self, is_train, dataloader_size, num_classes, storing_dict: dict):
        super(epoch_stats, self).__init__()
        self.is_train = is_train
        self.epoch_features = []
        self.epoch_outs = []
        self.epoch_train_grads = []
        self.epoch_labels = []
        self.running_corrects = 0
        self.running_loss = 0.0
        # self.running_st = 0.0
        # self.running_sw = 0.0
        # self.running_sb = 0.0
        self.epoch_loss = 0.0
        self.epoch_acc = 0.0
        self.c_var = []
        self.cross_cov = []
        self.dataloader_size = dataloader_size
        self.num_classes = num_classes
        self.mini_batch_count = 0
        self.storing_dict = storing_dict

    def update_stats_mini_batch(self, outs, loss_, aux_outputs, labels, preds, batch_indx):
        # if self.storing_dict['stats']: 
        #     st = loss_results_dict['st'].detach().cpu().numpy()
        #     sw = loss_results_dict['sw'].detach().cpu().numpy()
        #     sb = loss_results_dict['sb'].detach().cpu().numpy()
        #     self.running_st += st
        #     self.running_sb += sb
        #     self.running_sw += sb

        store_labels = False
        if (self.is_train and self.storing_dict['train_features']) or (not self.is_train and self.storing_dict['test_features']): 
            self.epoch_features = torch.clone(aux_outputs).cpu().detach().numpy() if batch_indx == 0 else np.vstack(
                (self.epoch_features, torch.clone(aux_outputs).cpu().detach().numpy()))
            
            store_labels = True

        if (self.is_train and self.storing_dict['train_outs']) or (not self.is_train and self.storing_dict['test_outs']): 
            self.epoch_outs = torch.clone(outs).cpu().detach().numpy() if batch_indx == 0 else np.vstack(
                (self.epoch_outs, torch.clone(outs).cpu().detach().numpy()))
            
            store_labels = True
           
        if store_labels: 
            self.epoch_labels = torch.clone(labels).cpu().detach().numpy() if batch_indx == 0 else np.concatenate(
                (self.epoch_labels, torch.clone(labels).cpu().detach().numpy()))
        
        

        self.running_loss += torch.clone(loss_).cpu().detach().numpy() * self.dataloader_size
        self.mini_batch_count += 1
        # print(batch_indx, ":", self.running_loss, loss_, self.dataloader_size)

        self.running_corrects += torch.sum(preds == labels)

    def update_stats_whole_batch(self):
        self.epoch_loss = (self.running_loss / self.dataloader_size)
        self.epoch_loss /= self.mini_batch_count

        
        self.epoch_acc = (self.running_corrects.double() / self.dataloader_size).detach().cpu().numpy()

        # calculate c_var (sw) [c, d]  and cross covariance for epoch_features:
        if self.storing_dict['covs']:
            self.c_var, self.cross_cov = get_c_var(self.epoch_features, self.epoch_labels, self.num_classes)
        else:
            self.c_var = None
            self.cross_cov = None

    def update_stored_grads(self, aux_outputs, batch_indx):
        if self.is_train and self.storing_dict['grads']:
            self.epoch_train_grads = torch.clone(aux_outputs.grad).cpu().detach().numpy() \
                if batch_indx == 0 else np.vstack(
                (self.epoch_train_grads, torch.clone(aux_outputs.grad).cpu().detach().numpy()))


class training_results_wrapper:

    def __init__(self, is_train, storing_dict: dict):
        super(training_results_wrapper, self).__init__()
        self.is_train = is_train
        self.features = []
        self.outs = []
        self.labels = []
        self.grads = []
        self.acc = []
        self.loss = []
        # self.st = []
        # self.sw = []
        # self.sb = []
        self.c_var = []
        self.cross_cov = []
        self.storing_dict = storing_dict

    def update(self, new_epoch_stats: epoch_stats):
        store_labels = False
        if (self.storing_dict['train_features'] and self.is_train) or (self.storing_dict['test_features'] and not self.is_train):
            self.features.append(new_epoch_stats.epoch_features)
            store_labels = True

        if (self.storing_dict['train_outs'] and self.is_train) or (self.storing_dict['test_outs'] and not self.is_train):
            self.outs.append(new_epoch_stats.epoch_outs)
            store_labels = True

        if store_labels: 
            self.labels.append(new_epoch_stats.epoch_labels)

        if self.is_train and self.storing_dict['grads']: 
            self.grads.append(new_epoch_stats.epoch_train_grads)
        
        self.acc.append(new_epoch_stats.epoch_acc)
        self.loss.append(new_epoch_stats.epoch_loss)

        # if self.storing_dict['stats']: 
        #     self.st.append(new_epoch_stats.running_st)
        #     self.sw.append(new_epoch_stats.running_sw)
        #     self.sb.append(new_epoch_stats.running_sb)
        
        if self.storing_dict['covs']: 
            self.c_var.append(new_epoch_stats.c_var)
            self.cross_cov.append(new_epoch_stats.cross_cov)

    def change_last_epoch_acc(self, new_acc):
        self.acc[-1] = new_acc

    def list_to_np_array(self):
        self.loss = np.array(self.loss)

        if self.storing_dict['acc']:
            self.acc = np.array(self.acc)

        if len(self.c_var) > 0:
            self.c_var = np.stack(self.c_var)
        else:
            self.c_var = np.array(self.c_var)

        if len(self.cross_cov) > 0:
            self.cross_cov = np.stack(self.cross_cov)
        else:
            self.cross_cov = np.array(self.cross_cov)

    def to_dict(self) -> dict:

        self.list_to_np_array()
        
        results_dict = {}
        # if self.storing_dict['stats']: 
        #     results_dict.update({
        #         'st': self.st,
        #         'sw': self.sw,
        #         'sb': self.sb,
        #     })
        results_dict['loss'] = self.loss
        if self.storing_dict['covs']: 
            results_dict.update({
                'c_var': self.c_var,
                'cross_cov': self.cross_cov
            })
        if self.storing_dict['acc']: 
            results_dict['acc'] = self.acc
        
        if self.is_train:            
            if self.storing_dict['grads']: 
                results_dict['grads'] = self.grads
            if self.storing_dict['train_outs']: 
                results_dict['outs'] = self.outs
            if self.storing_dict['train_features']: 
                results_dict['features'] = self.features
            if self.storing_dict['train_outs'] or self.storing_dict['train_features']:
                results_dict['labels'] = self.labels
    
        else:
            if self.storing_dict['test_outs']: 
                results_dict['outs'] = self.outs
            if self.storing_dict['test_features']: 
                results_dict['features'] = self.features
            if self.storing_dict['test_outs'] or self.storing_dict['test_features']:
                results_dict['labels'] = self.labels

        return results_dict

import torch 
def iterate_model(model: nn.Module, dataloader, device):
    model.eval()
    net_output = []
    y_output = []
    with torch.no_grad(): 

        for indx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            aux, outs = model(inputs)
            net_output.append(outs.cpu())
            y_output.append(labels.cpu())
            
    net_output = torch.vstack(net_output)
    y_output = torch.concatenate(y_output)

    
    return net_output, y_output

