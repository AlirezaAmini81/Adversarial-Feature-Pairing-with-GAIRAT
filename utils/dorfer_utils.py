import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.training_utils import training_results_wrapper, epoch_stats
from typing import List

def accuracy_score(t, p):
    """
    Compute accuracy
    """
    return float(np.sum(p == t)) / len(p)


class LDA(object):
    """ LDA Class """

    def __init__(self, r=1e-3, n_components=None, verbose=False, show=False):
        """ Constructor """
        self.r = r
        self.n_components = n_components

        self.scalings_ = None
        self.coef_ = None
        self.intercept_ = None
        self.means = None

        self.verbose = verbose
        self.show = show

    def fit(self, X, y, X_te=None, name_to_save: str = 'default_eig_plot.jpg'):
        """ Compute lda on hidden layer """

        # split into semi- and supervised- data
        # print('-'*20, ' inside LDA ', '-'*20)
        # print('X shape: ', X.shape)
        # print('y shape: ', y.shape)

        X_all = X.copy()
        X = X[y >= 0]
        y = y[y >= 0]

        # get class labels
        classes = np.unique(y)

        # set number of components
        if self.n_components is None:
            self.n_components = len(classes) - 1

        # compute means
        means = []
        for group in classes:
            Xg = X[y == group, :]
            means.append(Xg.mean(0))
        self.means = np.asarray(means)

        # compute covs
        covs = []
        for group in classes:
            Xg = X[y == group, :]
            Xg = Xg - np.mean(Xg, axis=0)
            covs.append(np.cov(Xg.T))

        # within scatter
        Sw = np.average(covs, axis=0)

        # total scatter
        X_all = X_all - np.mean(X_all, axis=0)
        if X_te is not None:
            St = np.cov(np.concatenate((X_all, X_te)).T)
        else:
            St = np.cov(X_all.T)

        # between scatter
        Sb = St - Sw

        # cope for numerical instability
        Sw += np.identity(Sw.shape[0]) * self.r

        # compute eigen decomposition
        from scipy.linalg.decomp import eigh
        evals, evecs = eigh(Sb, Sw)

        # sort eigen vectors according to eigen values
        evecs = evecs[:, np.argsort(evals)[::-1]]

        # normalize eigen vectors
        evecs /= np.apply_along_axis(np.linalg.norm, 0, evecs)

        # compute lda data
        self.scalings_ = evecs
        self.coef_ = np.dot(self.means, evecs).dot(evecs.T) # this is T
        self.intercept_ = (-0.5 * np.diag(np.dot(self.means, self.coef_.T)))

        if self.verbose:
            top_k_evals = evals[-self.n_components:]
            print("LDA-Eigenvalues (Train):",
                  np.array_str(top_k_evals, precision=2, suppress_small=True))
            print("Ratio min(eigval)/max(eigval): %.3f, Mean(eigvals): %.3f" %
                  (top_k_evals.min() / top_k_evals.max(), top_k_evals.mean()))

        if self.show:
            plt.figure("Eigenvalues")
            ax = plt.subplot(111)
            top_k_evals /= np.sum(top_k_evals)
            plt.plot(range(self.n_components), top_k_evals, 'bo-')
            plt.grid('on')
            plt.xlabel('Eigenvalue', fontsize=20)
            plt.ylabel('Explained Discriminative Variance', fontsize=20)
            plt.ylim([0.0, 1.05 * np.max(top_k_evals)])

            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            plt.savefig(f'{name_to_save}')
            plt.clf()

        return evals

    def transform(self, X):
        """ transform data """
        X_new = np.dot(X, self.scalings_)
        return X_new[:, :self.n_components]

    def predict_proba(self, X):
        """ estimate probability """
        prob = -(np.dot(X, self.coef_.T) + self.intercept_)
        np.exp(prob, prob) # store e^{prob} in prob
        prob += 1
        np.reciprocal(prob, prob)
        prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
        return prob

    def predict_log_proba(self, X):
        """ estimate log probability """
        return np.log(self.predict_proba(X))

    

def test_dorfer_optimal(curr_epoch_stats: epoch_stats, epoch_num: int, r, n_components=None):
    outs_total = np.array(curr_epoch_stats.epoch_features)
    y_total = np.array(curr_epoch_stats.epoch_labels)

    # fit lda on net output
    dlda = LDA(r=r, n_components=n_components, verbose=True)
    evals = dlda.fit(outs_total, y_total)

    # predict on train set
    proba = dlda.predict_proba(outs_total)
    # print('proba.shape shape: ', proba.shape)
    # print('proba.shape[0:10]: ', proba[0:10])

    # predicted class for each sample = class with maximum probabilty (probabilities were determined in .predict_proba() func)
    y_tr_pr = np.argmax(proba, axis=1)
    tr_acc = 100 * accuracy_score(y_total, y_tr_pr)

    if curr_epoch_stats.is_train: 
        print(f'Train Acc in epoch {epoch_num}: ', tr_acc)
    else:
        print(f'Test Acc in epoch {epoch_num}: ', tr_acc)

def unseen_data_classification_by_deepLDA(dlda: LDA, model_output, labels):
    # predict probabilties for train set
    proba = dlda.predict_proba(model_output)

    # assign class labels based on predicted probabilities
    pred_labels = np.argmax(proba, axis=1)
    train_acc = 100 * accuracy_score(labels, pred_labels)

    return pred_labels, train_acc


def classification_by_deepLDA(train_epoch_stats: epoch_stats, r, n_components=None): 
    """ return LDA, accuaracy """
    model_output = np.array(train_epoch_stats.epoch_features)
    labels = np.arrat(train_epoch_stats.epoch_labels)

    # fit LDA
    dlda = LDA(r, n_components=n_components, verbose=False)
    evals = dlda.fit(model_output, labels)

    pred_labels, train_acc = unseen_data_classification_by_deepLDA(dlda, model_output)
    
    return dlda, pred_labels, train_acc

def lda_acc(model_output, labels, epoch_num: int, r, n_components=None):
    outs_total = model_output
    y_total = labels

    # fit lda on net output
    dlda = LDA(r=r, n_components=n_components, verbose=False)
    evals = dlda.fit(outs_total, y_total)

    # predict on train set
    proba = dlda.predict_proba(outs_total)
    # print('proba.shape shape: ', proba.shape)
    # print('proba.shape[0:10]: ', proba[0:10])

    # predicted class for each sample = class with maximum probabilty (probabilities were determined in .predict_proba() func)
    y_tr_pr = np.argmax(proba, axis=1)
    return 100 * accuracy_score(y_total, y_tr_pr)

def eval_plots(features: List[np.ndarray], labels: List[np.ndarray], r, n_components=None):
    outs_total = np.array(features)
    y_total = np.array(labels)

    # compute lda on net outputs
    dlda = LDA(r=r, n_components=n_components, verbose=True, show=True)
    dlda.fit(outs_total, y_total)

    # predict on test set
    print("\nComputing accuracies ...")
    y_pr = np.argmax(dlda.predict_proba(outs_total), axis=1)
    print("LDA Accuracy on train set: %.3f" % (100 * accuracy_score(y_total, y_pr)))
    
    # project data to DeepLDA space
    XU_tr = dlda.transform(outs_total) # (n, 9)

    # scatter plot of projection components
    colors = plt.cm.jet(np.arange(0.0, 1.0, 0.1))[:, 0:3]
    plt.figure('DeepLDA-Feature-Scatter-Plot', facecolor='white')
    plt.clf()
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    plot_idx = 1
    for i in range(np.min([9, n_components])):
        for j in range(i + 1, np.min([9, n_components])):
            plt.subplot(6, 6, plot_idx)
            plot_idx += 1
            for l in np.unique(y_total):
                idxs = y_total == l
                plt.plot(XU_tr[idxs, i], XU_tr[idxs, j], 'o', color=colors[l], alpha=0.5)
            plt.axis('off')
            plt.axis('equal')

    # plot histograms of features
    plt.figure('DeepLDA-Feature-Histograms', facecolor='white')
    plt.clf()
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.0, hspace=0.0)
    for i in range(n_components):
        plt.subplot(5, 2, i + 1)

        F = XU_tr[:, i]
        min_val, max_val = F.min(), F.max()

        for c in np.unique(y_total):
            F = XU_tr[y_total == c, i]
            hist, bin_edges = np.histogram(F, range=(min_val, max_val), bins=100)
            plt.plot(bin_edges[1:], hist, '-', color=colors[c], linewidth=2)
        plt.axis('off')

    plt.show(block=True)
    