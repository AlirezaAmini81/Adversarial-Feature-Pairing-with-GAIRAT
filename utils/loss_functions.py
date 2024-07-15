import torch
import os
import sys

# relative import hacks (sorry)
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # for bash user
os.chdir(parentdir)  # for pycharm user


from utils.loss_function_utils import *
from torch.autograd.function import Function
import torch.nn as nn
import scipy.linalg as slinalg
from typing import Dict
from scipy.linalg import eigvalsh
from abc import ABC, abstractmethod
from utils.constants import Constants
import torch.nn.functional as F


class liliBaseClass(nn.Module, ABC):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        """
        Parameters
        ----------
            num_classes: int
                number of classes
            normalize_grad: bool
                if True, normalizes the aux_outputs gradients using register hook, such that
                aux_outputs.grad for every aux_output is in range [-1, 1]
        """
        super(liliBaseClass, self).__init__()
        self.num_classes = num_classes
        self.normalize_grad = normalize_grad
        self.loss_type = "lili"
        self.name: str

    @abstractmethod
    def forward(self, features: torch.Tensor, labels):
        # TODO normalizing grads when facing test set (which doesnt require grad)
        if self.normalize_grad and features.requires_grad:
            features.register_hook(lambda grad: grad / grad.abs().max())


# max(sw): max over dimension, sum over classes
class max_sw_trace_sb_2_Loss(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        """
        max_sw_trace_sb_2 returns the following loss:

                Σ max_diagonal(cov(h|c))                                     sum(max(sw_list))
        loss = ___________________________ (for every c \in num_classes) = ______________________
                    tr(cov(E[h|c]))                                            trace(sb)

        where h is the representation of a training sample: c is the corresponding class
        (E[h|c]: mean of class 'c'), max_diagonal returns maximum diagonal element of matrix

        *** for the numerator of this loss:                                 max over dim            sum
            max over dimensions, then sum over classes -> sw_list [c, d] -----------------> [c, ] -------> [1, ]

        Note: weighted Sw and Sb is used in this loss is weighted (get_scatters(weighted=True) by default)
        """
        super().forward(features, labels)
        st_list, sw_list, sb_list = get_scatters(features, labels, self.num_classes)
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        # max over dimension for each class, then sum over all classes
        sw_list_max = torch.max(sw_list, dim=1).values  # [c, ]
        sw = sw_list_max.sum()  # [1, ]

        sb = sb_list.sum()  # sum over dimensions [1, ]

        # unused in loss, only report st:

        loss = sw / sb

        return loss


# max(sw): sum over classes, max over dimensions
class max_sw_trace_sb_1_Loss(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        """
        max_sw_trace_sb_1 returns the following loss:

                max_diagonal(Σ(tr(cov(h|c))))                                    max(sum(sw_list))
        loss = ______________________________ (for every c \in num_classes) = ______________________
                    tr(cov(E[h|c]))                                                trace(sb)

        where h is the representation of a training sample: c is the corresponding class
        (E[h|c]: mean of class 'c'), max_diagonal returns maximum diagonal element of matrix

        *** for the numerator of this loss:                                  sum over classes            max
            sum over classes, then max over dimensions  -> sw_list [c, d] --------------------> [d, ] -------> [1, ]

        Note: weighted Sw and Sb is used in this loss is weighted (get_scatters(weighted=True) by default)
        """
        super().forward(features, labels)
        st_list, sw_list, sb_list = get_scatters(features, labels, self.num_classes)
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        # sum all variances (sum trace of scatter matrixes) , then max [1, ]
        sw_list_sum = sw_list.sum(dim=0)  # [d, ]
        sw = sw_list_sum.max()  # [1, ]

        sb = sb_list.sum()  # sum over dimensions [1, ]

        # unused in loss, only report st:
        st = st_list.sum()  # sum over dimensions and classes [1, ]
        if self.debug:
            print("sw: {:.4f}".format(sw.item()))
            print("sb: {:.4f}".format(sb.item()))
            print("st: {:.4f}".format(st.item()))

        loss = sw / sb

        return loss


# j1_1 (different interpretation of lili's j1 loss)
class j1_1_Loss(liliBaseClass):

    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        super().forward(features, labels)
        """
        j1_1 returns the following loss:

                        E[tr(cov(h|c))]
        loss = Σ ________________________________ (Σ c over num_classes, min over all classes except c)
                min(|| E[h|c] - E[h|c`] ||^2)

                                    E[tr(cov(h|c1))]                   E[tr(cov(h|c2))]
        in other words: loss = _______________________________ +  _______________________________ + ...
                                min(|| E[h|c1] - E[h|c`] ||^2)     min(|| E[h|c2] - E[h|c`] ||^2)

        where h is the representation of a training sample: c is the corresponding class
        (E[h|c]: mean of class 'c'), max_diagonal returns maximum diagonal element of matrix

        *** Note that "c" in numerator is the same "c" in the denominator

        Note: weighted Sw and Sb is used in this loss is weighted (get_scatters(weighted=True) by default)
        *** j1_1 uses weighted distance between centroids(means) (j1 does not uses weighted_dist_centroids)

        loss:
            - calculate the term "sw_list[c].sum() / minimum distance of class c with other classes" for each class c
            - loss = sum of the above term for all classes
        """
        st_list0, sw_list0, sb_list0 = get_scatters(features, labels, self.num_classes)
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        # dist1 = get_vect_dist_centroids(features, labels, self.num_classes)
        w_dist_centroids = get_dist_centroids(
            features, labels, self.num_classes, weighted=True
        )  # [c, c]

        loss = torch.tensor(0.0).to(sb_list0.get_device())
        for curr_c in range(self.num_classes):
            sw_curr_c = sw_list0[curr_c].sum()  # sum over dimnesions
            sb_curr_c = w_dist_centroids[curr_c]  # [c, ]
            sb_min_curr_c = (
                sb_curr_c.min()
            )  # minimum distance of class "curr_c" with other classes
            # print(sw_curr_c.get_device())
            # print(sb_min_curr_c.get_device())
            # print(loss.get_device())

            loss += sw_curr_c / sb_min_curr_c
            # loss += torch.log((sw_curr_c / sw_curr_c) + 1)

        # log ?
        loss = torch.log(loss + 1)

        return loss


# not optimized
class j1_Loss(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        """
        j1 is the reconstructuon of "vect_mc_discriminant_loss" implemented by LiLi.
        j1 returns the following loss:

                    mean of every E[tr(cov(h|c))] for every c in classes                                  mean(sw_list)
        loss = log __________________________________________________________ = log _______________________________________________________________________
                            min(|| E[h|c_k1] - E[h|c_k2] ||^2)                    distance of 2 classes with minimum pairwise distance among all classes

                                                1                        E[tr(cov(h|c1))] + E[tr(cov(h|c2))] + ... + E[tr(cov(h|c_num_classes))]
        in other words: loss = log( ___________________________________  ( ________________________________________________________________________ ))
                                    min(|| E[h|c_k1] - E[h|c_k2] ||^2)                                  num_classes

        where h is the representation of a training sample: c is the corresponding class
        (E[h|c]: mean of class 'c'), max_diagonal returns maximum diagonal element of matrix
        ***k1 and k2 are 2 classes with minimum pairwise distance among all classes

        *** Note that "c" in numerator is different from c_k1 and c_k2 in denominator

        Note: weighted Sw and Sb is used in this loss is weighted (get_scatters(weighted=True) by default)

        *** j1 does not uses weighted distance between centroids(means) (j1_1 uses weighted_dist_centroids)

        loss:
            - calculate mean(sw_list)
            - find minimum "un-weighted?" distance between all classes (class c_k1 and c_k2)
            - loss = mean(sw_list) / dist(c_k1, c_k2)
        """
        super().forward(features, labels)

        st_list0, sw_list0, sb_list0 = get_scatters(
            features, labels, self.num_classes, weighted=True
        )
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        dist_centroids = get_vect_dist_centroids(features, labels, self.num_classes)
        # dist_centroids = get_dist_centroids(features, labels, self.num_classes, weighted=False)  # [c, c]
        sw_mean = torch.mean(sw_list0)  # mean of classes and dimensions [1, ]
        min_dist = torch.min(dist_centroids)  # min distance between all classes [1, ]

        loss = torch.log((sw_mean / min_dist) + 1)

        return loss


class j1_2_Loss(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels, confusion_mat: torch.Tensor):
        """
        confusion_mat: [c, c]
        """
        super().forward(features, labels)
        # return None

        # cov_t, covs_w, cov_b = get_covs_unbalanced(features, labels, self.num_classes)

        st_list0, sw_list0, sb_list0 = get_scatters(
            features, labels, self.num_classes, weighted=True
        )
        sw_mean = torch.mean(sw_list0)  # mean of classes and dimensions [1, ]
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        dist_centroids = get_vect_dist_centroids(features, labels, self.num_classes)
        # min_dist = torch.min(dist_centroids)  # min distance between all classes [1, ]
        # select classes i, j based on confusion matrix:
        confusion_mat.fill_diagonal_(-1)
        i, j = (confusion_mat == torch.max(confusion_mat)).nonzero()[
            0
        ]  # if multiple maximum, pick first one
        print("conf i, j: ", i.item(), j.item())
        dist_i, dist_j = (dist_centroids == torch.min(dist_centroids)).nonzero()[0]
        print("dist i, j: ", dist_i.item(), dist_j.item(), end="\n---------\n")
        # print(dist_centroids)

        denom = dist_centroids[i, j]

        loss = torch.log((sw_mean / denom) + 1)

        return loss


class j1_3_Loss(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels, confusion_mat: torch.Tensor):
        """
        confusion_mat: [c, c]
        """
        super().forward(features, labels)

        mod_conf_mat = modified_conf_mat(confusion_mat)
        return None

        st_list0, sw_list0, sb_list0 = get_scatters(
            features, labels, self.num_classes, weighted=True
        )
        sw_list0 *= weights.unsqueeze(1).expand(-1, sw_list0.shape[1])
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]
        #### sw_mean = torch.mean(sw_list0)  # mean of classes and dimensions [1, ]
        # sum is enough, since weights are normalized:
        sw_mean = torch.sum(sw_list0)

        dist_centroids = get_vect_dist_centroids(features, labels, self.num_classes)
        # min_dist = torch.min(dist_centroids)  # min distance between all classes [1, ]

        dist_i, dist_j = (dist_centroids == torch.min(dist_centroids)).nonzero()[0]
        print("dist i, j: ", dist_i.item(), dist_j.item(), end="\n---------\n")
        # print(dist_centroids)

        denom = dist_centroids[i, j]

        loss = torch.log((sw_mean / denom) + 1)

        return loss


class w_j1_Loss(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

        print(
            "For testing purposes, sw in omitted and i=4 and j=9 is hardcoded in the forward function!!!"
        )

    def forward(self, features, labels, conf_mat, debug: bool = False):
        super().forward(features, labels)

        _, sw_list0, sb_list = get_scatters(
            features, labels, self.num_classes, weighted=True
        )
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]
        # print(conf_mat)
        mod_conf_mat = get_modified_conf_mat(conf_mat=conf_mat)
        weights = (mod_conf_mat.sum(1) / (mod_conf_mat.sum(1).sum() + 1e-12)).to(
            features.device
        )
        if weights.sum() == 0.0:
            weights = torch.ones_like(weights) / (weights.shape[0])
            # what i and j for denominator?

        # print('weights: ', np.array_str(weights.cpu().numpy(), precision=4, suppress_small=True))

        # print(torch.unique(labels, return_counts=True))

        #### Numerator
        sw_list0 *= weights.unsqueeze(1).expand(-1, sw_list0.shape[1])
        sw_mean = torch.sum(sw_list0)

        #### Denominator
        dist_centroids = get_vect_dist_centroids(features, labels, self.num_classes)

        # if debug == True:
        # print('conf_mat: ')
        # print(conf_mat)
        # print('dist mat')
        # print(dist_centroids)

        # print(dist_centroids)
        dist_i, dist_j = (dist_centroids == torch.min(dist_centroids)).nonzero()[0]
        # print('dist i, j: ', dist_i.item(), dist_j.item())
        # by distance matrix #
        # min_dist = torch.min(dist_centroids)  # min distance between all classes [1, ]

        # should i use the biggest element of conf_mat or mod_conf_mat?
        # probabaly mod_conf_mat

        # if acc of batch is 100%, it picks 0, 0
        # print(
        #     '-'*30
        # )
        # print(conf_mat)
        i, j = (mod_conf_mat == torch.max(mod_conf_mat)).nonzero()[
            0
        ]  # if multiple maximum, pick first one
        if debug:
            print(i, j)
        i = 4
        j = 9

        # print('conf i, j: ', i.item(), j.item())

        # https://pytorch.org/docs/stable/generated/torch.select.html#torch.select

        # by conf matrix #
        # TODO: check in test.py if indexing is differentalbe in pytorch
        if i != j:
            min_dist = dist_centroids[i, j]
        else:
            min_dist = sb_list.sum()

        # loss = torch.log((sw_mean / min_dist) + 1)
        loss = torch.log((1 / min_dist) + 1)

        # print(loss)

        return loss


def _cosine_distance(torch_vectors):
    """
    Generated by Chat-GPT,
    tested here: https://colab.research.google.com/drive/1UB8SxPE7SHZUYTK7xfl-Ji8bgiIenbwn?authuser=4#scrollTo=dCDE9q_CSHJ0
    """
    # Normalize vectors
    normalized_vectors = torch.nn.functional.normalize(torch_vectors, p=2, dim=1)

    # Compute dot product matrix
    dot_product = torch.matmul(normalized_vectors, normalized_vectors.t())

    # Clamp dot product to range [-1, 1] to avoid numerical errors
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Compute cosine similarity
    cosine_similarity = dot_product

    # Compute angular distance
    cosine_distance = 1 - cosine_similarity

    return cosine_distance


def cos_dist_centroids(features, labels, num_classes):
    means = []
    for i in range(num_classes):
        feats_c = features[labels == i]
        means.append(torch.mean(feats_c, dim=0)[None, :])
    means = torch.concatenate(means)

    return _cosine_distance(means)


class w_angular_j1(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__
        print(
            "For testing purposes, sw in omitted and i=4 and j=9 is hardcoded in the forward function!!!"
        )

    def forward(self, features, labels, conf_mat, debug: bool = False):
        super().forward(features, labels)

        _, sw_list0, sb_list = get_scatters(
            features, labels, self.num_classes, weighted=True
        )

        # print(conf_mat)
        mod_conf_mat = get_modified_conf_mat(conf_mat=conf_mat)
        weights = (mod_conf_mat.sum(1) / (mod_conf_mat.sum(1).sum() + 1e-12)).to(
            features.device
        )
        if weights.sum() == 0.0:
            weights = torch.ones_like(weights) / (weights.shape[0])
            # what i and j for denominator?

        #### Numerator
        sw_list0 *= weights.unsqueeze(1).expand(-1, sw_list0.shape[1])
        sw_mean = torch.sum(sw_list0)

        #### Denominator
        cosine_dist_centroids = cos_dist_centroids(features, labels, self.num_classes)

        dist_i, dist_j = (
            cosine_dist_centroids == torch.min(cosine_dist_centroids)
        ).nonzero()[0]

        i, j = (mod_conf_mat == torch.max(mod_conf_mat)).nonzero()[
            0
        ]  # if multiple maximum, pick first one
        i = 4
        j = 9

        if debug:
            print(i, j)

        # https://pytorch.org/docs/stable/generated/torch.select.html#torch.select
        # by conf matrix #
        # TODO: check in test.py if indexing is differentalbe in pytorch
        if i != j:
            min_dist = cosine_dist_centroids[i, j]
        else:
            min_dist = sb_list.sum()

        # loss = torch.log((sw_mean / min_dist) + 1)
        loss = torch.log((1 / (min_dist + Constants.eta)) + 1)

        return loss


class w_angular_j1_v1(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__
        print(
            "The indices of the smallest element of cosine_dist_centroids matrix are picked as i and j"
        )

    def forward(self, features, labels, conf_mat, debug: bool = False):
        super().forward(features, labels)

        _, sw_list0, sb_list = get_scatters(
            features, labels, self.num_classes, weighted=True
        )
        #### Denominator
        cosine_dist_centroids = cos_dist_centroids(
            features, labels, self.num_classes
        ).fill_diagonal_(float("Inf"))

        i, j = (cosine_dist_centroids == torch.min(cosine_dist_centroids)).nonzero()[0]

        if debug:
            print(i, j)

        # https://pytorch.org/docs/stable/generated/torch.select.html#torch.select
        # by conf matrix #
        # TODO: check in test.py if indexing is differentalbe in pytorch
        if i != j:
            min_dist = cosine_dist_centroids[i, j]
        else:
            min_dist = sb_list.sum()

        loss = torch.log((1 / (min_dist + Constants.eta)) + 1)
        return loss


class w_angular_j1_v1_sw(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__
        print(
            "The indices of the smallest element of cosine_dist_centroids matrix are picked as i and j"
        )

    def forward(self, features, labels, conf_mat, debug: bool = False):
        super().forward(features, labels)

        _, sw_list0, sb_list = get_scatters(
            features, labels, self.num_classes, weighted=True
        )

        # mod_conf_mat = get_modified_conf_mat(conf_mat=conf_mat)
        # weights = (mod_conf_mat.sum(1) / (mod_conf_mat.sum(1).sum() + 1e-12)).to(
        #     features.device
        # )
        # if weights.sum() == 0.0:
        #     weights = torch.ones_like(weights) / (weights.shape[0])
        #     # what i and j for denominator?
        # #### Numerator
        # sw_list0 *= weights.unsqueeze(1).expand(-1, sw_list0.shape[1])
        sw_mean = torch.sum(sw_list0)

        #### Denominator
        cosine_dist_centroids = cos_dist_centroids(
            features, labels, self.num_classes
        ).fill_diagonal_(float("Inf"))

        i, j = (cosine_dist_centroids == torch.min(cosine_dist_centroids)).nonzero()[0]

        if debug:
            print(i, j)

        # https://pytorch.org/docs/stable/generated/torch.select.html#torch.select
        # by conf matrix #
        # TODO: check in test.py if indexing is differentalbe in pytorch
        if i != j:
            min_dist = cosine_dist_centroids[i, j]
        else:
            min_dist = sb_list.sum()

        loss = torch.log((sw_mean / (min_dist + Constants.eta)) + 1)
        return loss


class w_angular_j1_v2(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__
        print(
            "The indices of the biggest element of confusion_matrix are picked as i and j"
        )

    def forward(self, features, labels, conf_mat, debug: bool = False):
        super().forward(features, labels)

        _, sw_list0, sb_list = get_scatters(
            features, labels, self.num_classes, weighted=True
        )

        # print(conf_mat)
        mod_conf_mat = get_modified_conf_mat(conf_mat=conf_mat)
        weights = (mod_conf_mat.sum(1) / (mod_conf_mat.sum(1).sum() + 1e-12)).to(
            features.device
        )
        if weights.sum() == 0.0:
            weights = torch.ones_like(weights) / (weights.shape[0])
            # what i and j for denominator?

        #### Denominator
        cosine_dist_centroids = cos_dist_centroids(features, labels, self.num_classes)

        i, j = (mod_conf_mat == torch.max(mod_conf_mat)).nonzero()[
            0
        ]  # if multiple maximum, pick first one

        if debug:
            print(i, j)

        # https://pytorch.org/docs/stable/generated/torch.select.html#torch.select
        # by conf matrix #
        # TODO: check in test.py if indexing is differentalbe in pytorch
        if i != j:
            min_dist = cosine_dist_centroids[i, j]
        else:
            min_dist = sb_list.sum()

        loss = torch.log((1 / (min_dist + Constants.eta)) + 1)
        return loss


class w_angular_lda_v1(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__
        print(
            "For testing purposes, i=4 and j=9 is hardcoded in the forward function!!!"
        )

    def forward(self, features, labels, conf_mat, debug: bool = False):
        super().forward(features, labels)

        _, sw_list0, sb_list = get_scatters(
            features, labels, self.num_classes, weighted=True
        )

        # print(conf_mat)
        mod_conf_mat = get_modified_conf_mat(conf_mat=conf_mat)
        # weights = (mod_conf_mat.sum(1) / (mod_conf_mat.sum(1).sum()+1e-12)).to(features.device)
        # if weights.sum() == 0.0:
        #     weights = torch.ones_like(weights) / (weights.shape[0])
        #     # what i and j for denominator?

        #### Numerator
        # sw_list0 *= weights.unsqueeze(1).expand(-1, sw_list0.shape[1])
        sw_mean = torch.sum(sw_list0)

        #### Denominator
        cosine_dist_centroids = cos_dist_centroids(features, labels, self.num_classes)

        # print(cosine_dist_centroids)
        dist_i, dist_j = (
            cosine_dist_centroids == torch.min(cosine_dist_centroids)
        ).nonzero()[0]

        i, j = (mod_conf_mat == torch.max(mod_conf_mat)).nonzero()[
            0
        ]  # if multiple maximum, pick first one
        i = 4
        j = 9

        if debug:
            print(i, j)

        # https://pytorch.org/docs/stable/generated/torch.select.html#torch.select
        # by conf matrix #
        # TODO: check in test.py if indexing is differentalbe in pytorch
        if i != j:
            min_dist = cosine_dist_centroids[i, j]
        else:
            min_dist = sb_list.sum()

        # loss = torch.log((sw_mean / min_dist) + 1)
        loss = torch.log((sw_mean / (min_dist + Constants.eta)) + 1)

        return loss


class w_angular_lda_v2(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__
        print(
            "For testing purposes, i=4 and j=9 is hardcoded in the forward function!!!"
        )

    def forward(self, features, labels, conf_mat, debug: bool = False):
        super().forward(features, labels)

        _, sw_list0, sb_list = get_scatters(
            features, labels, self.num_classes, weighted=True
        )

        # print(conf_mat)
        mod_conf_mat = get_modified_conf_mat(conf_mat=conf_mat)
        # weights = (mod_conf_mat.sum(1) / (mod_conf_mat.sum(1).sum()+1e-12)).to(features.device)
        # if weights.sum() == 0.0:
        #     weights = torch.ones_like(weights) / (weights.shape[0])
        #     # what i and j for denominator?

        #### Numerator
        # sw_list0 *= weights.unsqueeze(1).expand(-1, sw_list0.shape[1])
        # sw_mean = torch.sum(sw_list0)
        sw_mean = torch.sum(sw_list0[4] + sw_list0[9])

        #### Denominator
        cosine_dist_centroids = cos_dist_centroids(features, labels, self.num_classes)

        # print(cosine_dist_centroids)
        dist_i, dist_j = (
            cosine_dist_centroids == torch.min(cosine_dist_centroids)
        ).nonzero()[0]

        i, j = (mod_conf_mat == torch.max(mod_conf_mat)).nonzero()[
            0
        ]  # if multiple maximum, pick first one
        i = 4
        j = 9

        if debug:
            print(i, j)

        # https://pytorch.org/docs/stable/generated/torch.select.html#torch.select
        # by conf matrix #
        # TODO: check in test.py if indexing is differentalbe in pytorch
        if i != j:
            min_dist = cosine_dist_centroids[i, j]
        else:
            min_dist = sb_list.sum()

        # loss = torch.log((sw_mean / min_dist) + 1)
        loss = torch.log((sw_mean / (min_dist + Constants.eta)) + 1)

        return loss


class w_j1_Loss_v2(liliBaseClass):
    """
    wj1 -> only Sw, weighted by confusion matrix | no denom...
    """

    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels, conf_mat, debug: bool = False):
        super().forward(features, labels)

        _, sw_list0, _ = get_scatters(features, labels, self.num_classes, weighted=True)
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        # print(conf_mat)
        mod_conf_mat = get_modified_conf_mat(conf_mat=conf_mat)
        weights = (mod_conf_mat.sum(1) / (mod_conf_mat.sum(1).sum() + 1e-12)).to(
            features.device
        )
        if weights.sum() == 0.0:
            weights = torch.ones_like(weights) / (weights.shape[0])

        #### Numerator
        sw_list0 *= weights.unsqueeze(1).expand(-1, sw_list0.shape[1])
        sw_mean = torch.sum(sw_list0)

        return sw_mean


class sw_alone(liliBaseClass):
    """
    sw_alone -> only Sw, no weight
    """

    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels, debug: bool = False):
        super().forward(features, labels)

        _, sw_list0, _ = get_scatters(features, labels, self.num_classes, weighted=True)
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        sw_mean = torch.mean(sw_list0)

        return sw_mean


class sw_alone_log(liliBaseClass):
    """
    sw_alone -> only Sw, no weight
    """

    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels, debug: bool = False):
        super().forward(features, labels)

        _, sw_list0, _ = get_scatters(features, labels, self.num_classes, weighted=True)
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        sw_mean = torch.mean(sw_list0)

        return torch.log(sw_mean + 1)


class w_sw_alone_log(liliBaseClass):
    """
    sw_alone -> only Sw, no weight
    """

    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels, conf_mat, debug: bool = False):
        super().forward(features, labels)

        _, sw_list0, _ = get_scatters(features, labels, self.num_classes, weighted=True)
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        mod_conf_mat = get_modified_conf_mat(conf_mat=conf_mat)
        weights = (mod_conf_mat.sum(1) / (mod_conf_mat.sum(1).sum() + 1e-12)).to(
            features.device
        )
        if weights.sum() == 0.0:
            weights = torch.ones_like(weights) / (weights.shape[0])

        sw_list0 *= weights.unsqueeze(1).expand(-1, sw_list0.shape[1])
        sw_mean = torch.sum(sw_list0)

        return torch.log(sw_mean + 1)


class w_j1_Loss_v3(liliBaseClass):
    """
    Weighted Sb in the denominator,
    no use of distance matrix...
    """

    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels, conf_mat, debug: bool = False):
        super().forward(features, labels)

        st, sw_list0, sb_list = get_scatters(
            features, labels, self.num_classes, weighted=True
        )
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        # print(conf_mat)
        mod_conf_mat = get_modified_conf_mat(conf_mat=conf_mat)
        weights = (mod_conf_mat.sum(1) / (mod_conf_mat.sum(1).sum() + 1e-12)).to(
            features.device
        )
        if weights.sum() == 0.0:
            weights = torch.ones_like(weights) / (weights.shape[0])
            # what i and j for denominator?

        #### Numerator
        sw_list0 *= weights.unsqueeze(1).expand(-1, sw_list0.shape[1])  # [d, ]
        sw_mean = torch.sum(sw_list0)

        sb = st - sw_list0
        denom = sb.sum()

        loss = torch.log((sw_mean / denom) + 1)

        return loss


class j3_Loss(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        """ """
        super().forward(features, labels)

        st_list0, sw_list0, sb_list0 = get_scatters(
            features, labels, self.num_classes, weighted=True
        )
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]
        # max over dimension for each class, then sum over all classes
        sw_list_max = torch.max(sw_list0, dim=1).values  # [c, ]
        sw = sw_list_max.sum()  # [1, ]

        # dist_centroids = get_vect_dist_centroids(features, labels, num_classes)
        dist_centroids = get_dist_centroids(
            features, labels, self.num_classes, weighted=False
        )  # [c, c]
        # sw_mean = torch.mean(sw_list0)  # mean of classes and dimensions [1, ]
        min_dist = torch.min(dist_centroids)  # min distance between all classes [1, ]

        loss = torch.log((sw / min_dist) + 1)

        return loss


class vect_mc_discriminant_Loss(liliBaseClass):
    """
    LiLi's VECTORIZED implementation of j1 loss,
    """

    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        super().forward(features, labels)

        # c = torch.unique(labels, sorted=True)
        num_sample = torch.tensor(len(features)).type("torch.FloatTensor")

        one_hot_y = torch.zeros(len(labels), labels.max() + 1).scatter_(
            1, labels.cpu().unsqueeze(1), 1.0
        )
        # [N, C]
        # [i, :] is [0, 0, 0, 1, 0, 0] when i-th sample belongs to class 3 (and C = 6)

        # H = torch.eye(features.size()[0]) - torch.ones(features.size()[0]) / num_sample
        if features.is_cuda:
            one_hot_y = one_hot_y.type("torch.cuda.FloatTensor")
            # H = H.type('torch.cuda.FloatTensor')

        c_num_sample = torch.mm(
            one_hot_y.t(), one_hot_y
        )  # [c,n] x [n,c], number of samples in each class sitting on diagno
        try:
            c_num_inv = torch.inverse(
                c_num_sample
            )  # with shape [c,c], preparing for average
        except:
            print("sample zero")
            return None

        # centered_x = torch.mm(H, features)
        conditional_c_add = torch.mm(
            one_hot_y.t(), features
        )  # [c,n] x [n,d], sum of the samples condition on class
        c_centroids = torch.mm(
            c_num_inv, conditional_c_add
        )  # [c,c] x [c,d], conditional means

        power2_x = torch.pow(features, 2)
        power2_conditional_c_add = torch.mm(
            one_hot_y.t(), power2_x
        )  # [c,n] x [n,d], sum of the squared samples condition on class
        power2_centroids = torch.mm(
            c_num_inv, power2_conditional_c_add
        )  # [c,c] x [c,d], conditional means of x.^2

        c_var = power2_centroids - torch.pow(
            c_centroids, 2
        )  # [c, d], conditional variance for classes
        # Var [X] = E[X^2] - E[X]^2

        c_prob = (
            torch.diag(c_num_sample) / num_sample
        )  # [c,], probability of the classes appeared in a batch

        # dist_centroids is a 2d matrix
        dist_centroids = torch.norm(
            c_centroids[:, None] - c_centroids, dim=2, p=2
        )  # [c, c], symmetric. This is the trick part of computing the
        # pairewise distance between the centroids of classes.

        dist_centroids = torch.pow(
            dist_centroids, 2
        )  # [c, c], symmetric. Norm 2 squared

        # Find the categories which have the smallest centroids distance
        dist_centroids = torch.triu(
            dist_centroids, diagonal=1
        )  # [c, c], truncate half of the symmetric matrix.
        dist_centroids[dist_centroids == 0] = float(
            "Inf"
        )  # [c, c], set self distance to infinite.
        # print(dist_centroids)
        # dim0_min_val, dim0_min_idx = torch.min(dist_centroids.detach(),dim=0)
        # dim1_min_idx = dim0_min_val.argmin()

        # min_sub = (dim0_min_idx[dim1_min_idx].item(), dim1_min_idx.item())
        # min_dist = dist_centroids[min_sub]
        # print(min_dist)

        min_dist = torch.mean(
            torch.topk(dist_centroids.view(1, -1), k=1, largest=False)[0]
        )  # [1,] top-k smallest distance of the class centroids.
        # print('c_var', c_var)
        c_var_expect = torch.mean(
            c_var * c_prob[:, None]
        )  # [1,] expectation of the conditional variance of classes
        # print("c_var_exp: ", c_var_expect)
        loss = torch.log((c_var_expect / min_dist) + 1)  # [1,]

        return loss


class j2_1_Loss(liliBaseClass):

    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        """
        j2_1 returns the following loss:

                        E[tr(cov(h|c))]                           sw_list[c].sum()
        loss = max ______________________________ = __________________________________________________ (where class "c" maximizes this term)
                c   min(|| E[h|c] - E[h|c`] ||^2)     minimum distance of class "c" with other classes

        where h is the representation of a training sample: c is the corresponding class
        (E[h|c]: mean of class 'c'), max_diagonal returns maximum diagonal element of matrix

        *** Note that "c" in numerator is the same as c in denominator

        Note: weighted Sw and Sb is used in this loss is weighted (get_scatters(weighted=True) by default)
            *** original j2 does not use weighted Sw (get_scatters(weighted=False))

        *** j2_1 uses weighted distance between centroids(means) (j2 does not uses weighted_dist_centroids)

        loss:
            - for each class "c", calculate the term "sw_list[c].sum() / minimum distance of class "c" with other classes"
            - class "c" is the class that maximizes the above term,
            - loss = sw_list[c].sum() / minimum distance of class "c" with other classes
        """
        super().forward(features, labels)

        st_list0, sw_list0, sb_list0 = get_scatters(features, labels, self.num_classes)
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        # dist1 = get_vect_dist_centroids(features, labels, self.num_classes)
        w_dist_centroids = get_dist_centroids(
            features, labels, self.num_classes, weighted=True
        )  # [c, c]

        loss = torch.tensor(-1)
        for curr_c in range(self.num_classes):
            sw_curr_c = sw_list0[curr_c].sum()  # sum over dimnesions
            sb_curr_c = w_dist_centroids[curr_c]  # [c, ]
            sb_min_curr_c = sb_curr_c.min()

            loss_term_curr_c = sw_curr_c / sb_min_curr_c
            # maximize
            if loss < loss_term_curr_c:
                loss = loss_term_curr_c

        return loss


class j2_Loss(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        """
        j2 is the reconstruction of "vect_minimax_disc_loss" implemented by LiLi.
        j2 returns the following loss:

                    max E[tr(cov(h|c_k1))]                           sw_list[c_k1].sum()
        loss =  ___________________________________ = ______________________________________________________ (where class "c_k1" maximizes THE NUMERATOR)
                min(|| E[h|c_k1] - E[h|c`] ||^2)       minimum distance of class "c_k1" with other classes

        where h is the representation of a training sample: c is the corresponding class
        (E[h|c]: mean of class 'c'), max_diagonal returns maximum diagonal element of matrix

        *** Note that "c_k1" in numerator is the same as c_k1 in denominator

        Note: Sw used to calculate the numerator is not weighted (get_scatters(weighted=False)
            dist_centroids is not weighted as well.

        loss:
            - find the class that has the maximum sum of variances (sum over dimens) (** not weighted variances) --> class c_k1
            - find the closest class with c_k1 --> c` (** not weighted distance)
            - loss = sum of variances of c_k1 / distance[c_k1, c`]
        """
        super().forward(features, labels)

        st_list0, sw_list0, sb_list0 = get_scatters(
            features, labels, self.num_classes, weighted=False
        )
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        # dist_centroids = get_vect_dist_centroids(features, labels, self.num_classes)
        dist_centroids = get_dist_centroids(
            features, labels, self.num_classes, weighted=False
        )  # [c, c]

        sw_list_sum_over_dims = sw_list0.sum(dim=1)  # [c, ] sum over dims
        # print(sw_list0)
        max_sw_value, max_sw_indx = torch.max(sw_list_sum_over_dims, dim=0)

        min_dist_from_max_sw_class = dist_centroids[max_sw_indx].min()
        loss = max_sw_value / min_dist_from_max_sw_class

        return loss


class vect_minimax_disc_Loss(liliBaseClass):
    """
    LiLi's vectorized implementation of j2 loss
    """

    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        super().forward(features, labels)

        # num_sample = torch.tensor(len(features)).type('torch.FloatTensor')
        one_hot_y = torch.zeros(len(labels), labels.max() + 1).scatter_(
            1, labels.cpu().unsqueeze(1), 1.0
        )
        # H = torch.eye(features.size()[0]) - torch.ones(features.size()[0]) / num_sample
        if features.is_cuda:
            one_hot_y = one_hot_y.type("torch.cuda.FloatTensor")
            # H = H.type('torch.cuda.FloatTensor')

        c_num_sample = torch.mm(
            one_hot_y.t(), one_hot_y
        )  # [c,n] x [n,c], number of samples in each class sitting on diagno
        try:
            c_num_inv = torch.inverse(
                c_num_sample
            )  # with shape [c,c], preparing for average
        except:
            print("sample zero")
            return None

        # centered_x = torch.mm(H, features)
        conditional_c_add = torch.mm(
            one_hot_y.t(), features
        )  # [c,n] x [n,d], sum of the samples condition on class
        c_centroids = torch.mm(
            c_num_inv, conditional_c_add
        )  # [c,c] x [c,d], conditional means

        power2_x = torch.pow(features, 2)
        power2_conditional_c_add = torch.mm(
            one_hot_y.t(), power2_x
        )  # [c,n] x [n,d], sum of the squared samples condition on class
        power2_centroids = torch.mm(
            c_num_inv, power2_conditional_c_add
        )  # [c,c] x [c,d], conditional means of x.^2

        c_var = power2_centroids - torch.pow(
            c_centroids, 2
        )  # [c, d], conditional variance for classes
        c_var_sum = torch.sum(c_var, dim=1, keepdim=True)
        # print(c_var)
        c_var_max_val, c_var_max_idx = torch.max(c_var_sum, dim=0)
        c_var_max_val = c_var_max_val[0]
        c_var_max_idx = c_var_max_idx.item()

        dist_centroids = torch.norm(
            c_centroids - c_centroids[c_var_max_idx], p=2, dim=1
        )
        min_dist, _ = torch.sort(dist_centroids)
        min_dist_powered = torch.pow(min_dist[1], 2)
        loss = c_var_max_val / min_dist_powered
        # loss = torch.log(c_var_max_val / min_dist_powered + 1)

        return loss

class CL_Base(nn.Module, ABC):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.name = self.__class__.__name__
        self.loss_type = "CL_Base" 
        self.num_classes = num_classes
        self.feat_dim = feat_dim
    
    @abstractmethod
    def forward(self):
        pass

# Base Class of CL Losses that the centers are the same as the weights
class CL_weights(CL_Base, ABC):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__(num_classes, feat_dim)
    
    @abstractmethod
    def forward(self):
        pass

# Base Class of CL Losses that the centers are separate from the weights
class CL_not_weights(CL_Base, ABC):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__(num_classes, feat_dim)
    
    @abstractmethod
    def forward(self):
        pass


class CenterLoss(CL_not_weights):
    """
    https://github.com/jxgu1016/MNIST_center_loss_pytorch
    """
    def __init__(self, num_classes, feat_dim, center_loss_lr=0.5, beta=1.0, size_average=True):
        """
        Parameters
        ----------
            num_classes: int
                number of classes
            feat_dim: int
                feature's dimension
            beta: float
                beta is the coefficient used to multiply thisLoss and sum it up with nllloss to get the total loss. (loss weight)
                total_loss = NLLLoss(outs) + beta * thisLoss(aux_outs)
            center_loss_lr: float
                learning rate for updating centers
            size_average: bool
                author mentioned use of this parameter in github page. no use for us.
        """
        super().__init__(num_classes, feat_dim)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.size_average = size_average
        self.center_loss_lr = center_loss_lr

        if beta == 0:
            self.name = "SoftmaxLoss"
        else:
            self.name = self.__class__.__name__

        self.num_classes = num_classes
        self.beta = beta
        self.loss_type = "CL"

    def forward(self, features, labels, grad=True) -> Dict[str, torch.Tensor]:
        """
        m = batch_size
        label - [m, ]
        feat - [m, dim] - in case of mnist toy example: [m, 2]
        """
        batch_size = features.size(0)
        features = features.view(batch_size, -1)
        # To check the dim of centers and features
        if features.size(1) != self.feat_dim:
            raise ValueError(
                "Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(
                    self.feat_dim, features.size(1)
                )
            )

        batch_size_tensor = features.new_empty(1).fill_(
            batch_size if self.size_average else 1
        )

        if grad:
            loss = self.centerlossfunc(
                features, labels, self.centers, batch_size_tensor
            )
        else:
            loss = self.centerlossfunc(
                features, labels, self.centers.clone().detach(), batch_size_tensor
            )
        return loss


"""
Function
CLASStorch.autograd.Function(*args, **kwargs)[SOURCE]
Base class to create custom autograd.Function

To create a custom autograd.Function, subclass this class and implement the forward() 
and backward() static methods. Then, to use your custom op in the forward pass, 
call the class method apply. Do not call forward() directly.

To ensure correctness and best performance, make sure you are calling 
the correct methods on ctx and validating your backward function using 
torch.autograd.gradcheck().
"""


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        """
        torch.index_select(input, dim, index, *, out=None):

        Returns a new tensor which indexes the input tensor along 
        dimension dim using the entries in index which is a LongTensor.

        The returned tensor has the same number of dimensions as the original tensor (input).
        The dimth dimension has the same size as the length of index;
        other dimensions have the same size as in the original tensor.
        """
        centers_batch = centers.index_select(0, label.long())  # [m, feat_dim]
        # centers_batch[i, :] = i-th input's corresponding center (c_yi)
        # feature - centers_batch = x_i - c_yi
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())  # [m, feat_dim]
        # centers_batch[i, :] = i-th input's corresponding center (c_yi)

        diff = centers_batch - feature  # c_yi - x_i
        # init every iteration
        counts = centers.new_ones(centers.size(0))  # [c, ] of ones
        ones = centers.new_ones(label.size(0))  # [m, ] of ones
        grad_centers = centers.new_zeros(centers.size())  # [c, feat_dim] of zeros

        # 1-D case:
        # grad_centers[label[i]] += ones[i]
        counts = counts.scatter_add_(0, label.long(), ones)

        """
        Tensor.scatter_add_(dim, index, src)
        For a 3-D tensor, self is updated as:
            self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2
        The backward pass is implemented only for src.shape == index.shape
        """

        # dim = 0
        # index:
        # label.unsqueeze(1) changes the shape of label from [m, ] to [m, 1]
        # .expand(feature.size()): feature.size()=[m, feat_dim] --> label=[m, feat_dim] (Expanded horizontally)
        # label = [3, 4, 9, 1, 2]
        # index = [[3, 3],
        #          [4, 4],
        #          [9, 9],
        #          [1, 1],
        #          [2, 2]]

        # src = diff [m, feat_dim] , its c_yi - x_i

        # 2-D case:
        # grad_centers[label[i]][j] += diff[i][j]
        # :
        # grad_centers[c_yi][0] += c_yi - x_i[0]
        # grad_centers[c_yi][1] += c_yi - x_i[1]
        grad_centers.scatter_add_(
            0, label.unsqueeze(1).expand(feature.size()).long(), diff
        )  # [c, feat_dim]

        grad_centers = grad_centers / counts.view(-1, 1)

        return -grad_output * diff / batch_size, None, grad_centers / batch_size, None


class deepLDA(nn.Module):
    def __init__(
        self,
        num_classes,
        r=1e-3,
        n_components: int = None,
        epsilon: float = 1.0,
        normalize_grad=False,
    ):
        super().__init__()
        self.name = self.__class__.__name__
        self.loss_type = "dorfer"  # ؟‌
        self.num_classes = num_classes

        self.epsilon = epsilon
        self.r = r
        if n_components is None:
            self.n_components = num_classes - 1
        else:
            self.n_components = n_components

        self.normalize_grad = normalize_grad

        print(self.n_components)
        print(self.r)

    def forward(self, features, labels, debug=False):
        """
        m = batch_size
        label - [m, ]
        feat - [m, dim] - in case of mnist toy example: [m, 2]
        """
        # dump = [features.cpu().detach().numpy(), labels.cpu().detach().numpy()]
        # with open('features_labels.pickle', 'wb') as handle:
        #     pickle.dump(dump, handle, protocol=1)

        # TODO normalizing grads when facing test set (which doesnt require grad)
        if self.normalize_grad and features.requires_grad:
            features.register_hook(lambda grad: grad / grad.abs().max())

        St_t, Sw_t, _ = get_covs(features, labels, self.num_classes)
        Sw_t = torch.mean(Sw_t, dim=0)  # from (c, d, d) to (d, d)

        # just like DORFER:
        Sb_t = St_t - Sw_t

        # if debug:
        #     print('st')
        #     print(St_t)
        #     print('sw')
        #     print(Sw_t)
        #     print('sb')
        #     print(Sb_t)

        # print('rank(St): ', torch.linalg.matrix_rank(St_t).item(), '\t', St_t.shape)
        # print('rank(Sw): ', torch.linalg.matrix_rank(Sw_t).item(), '\t', Sw_t.shape)
        # print('rank(Sb): ', torch.linalg.matrix_rank(Sb_t).item(), '\t', Sb_t.shape)

        # try to calculate Sw t b just like what dorfer did:
        # https://github.com/CPJKU/deep_lda/blob/e47755bae54a5cb33f59c3954aaefd6f84b91cd0/deeplda/models/mnist_dlda.py#L119

        if Sw_t.device != torch.device("cpu"):
            Sw_t += torch.eye(Sw_t.shape[0]).to(Sw_t.get_device()) * self.r
        else:
            Sw_t += torch.eye(Sw_t.shape[0]) * self.r

        eig_prob_mat = torch.matmul(torch.linalg.inv(Sw_t), Sb_t)

        evals_t = torch.linalg.eigvals(eig_prob_mat)

        evals_t, _ = torch.sort(evals_t.real)  # ascending

        top_n_comp_evals = evals_t[
            -self.n_components :
        ]  # get top n_components (in general case, n_Comp = C-1)

        thresh = torch.min(top_n_comp_evals) + self.epsilon  # epsilon = 1

        top_k_evals = top_n_comp_evals[top_n_comp_evals <= thresh]
        # print('k in dorfer loss: ', top_k_evals.shape[0])
        # print('selected eigvals: ', np.array_str(top_k_evals.detach().cpu().numpy(), precision=3, suppress_small=True))
        if debug:
            print(
                "k: ",
                top_k_evals.shape[0],
                np.array_str(
                    evals_t.detach().cpu().numpy(), precision=3, suppress_small=True
                ),
            )

        loss = -torch.mean(top_k_evals)

        return loss


class deepLDA_v2(nn.Module):
    def __init__(
        self,
        num_classes,
        r=1e-3,
        n_components: int = None,
        epsilon: float = 1.0,
        normalize_grad=False,
    ):
        super().__init__()
        self.name = self.__class__.__name__
        self.loss_type = "dorfer"  # ؟‌
        self.num_classes = num_classes

        self.epsilon = epsilon
        self.r = r
        if n_components is None:
            self.n_components = num_classes - 1
        else:
            self.n_components = n_components

        self.normalize_grad = normalize_grad

        print(self.n_components)
        print(self.r)

    def forward(self, features, labels, debug=False):
        """
        m = batch_size
        label - [m, ]
        feat - [m, dim] - in case of mnist toy example: [m, 2]
        """
        # dump = [features.cpu().detach().numpy(), labels.cpu().detach().numpy()]
        # with open('features_labels.pickle', 'wb') as handle:
        #     pickle.dump(dump, handle, protocol=1)

        # TODO normalizing grads when facing test set (which doesnt require grad)
        if self.normalize_grad and features.requires_grad:
            features.register_hook(lambda grad: grad / grad.abs().max())

        St_t, Sw_t, Sb_t = get_covs_unbalanced(features, labels, self.num_classes)
        Sw_t = torch.mean(Sw_t, dim=0)  # from (c, d, d) to (d, d)

        # if debug:
        #     print('st')
        #     print(St_t)
        #     print('sw')
        #     print(Sw_t)
        #     print('sb')
        #     print(Sb_t)

        # print('rank(St): ', torch.linalg.matrix_rank(St_t).item(), '\t', St_t.shape)
        # print('rank(Sw): ', torch.linalg.matrix_rank(Sw_t).item(), '\t', Sw_t.shape)
        # print('rank(Sb): ', torch.linalg.matrix_rank(Sb_t).item(), '\t', Sb_t.shape)

        # try to calculate Sw t b just like what dorfer did:
        # https://github.com/CPJKU/deep_lda/blob/e47755bae54a5cb33f59c3954aaefd6f84b91cd0/deeplda/models/mnist_dlda.py#L119

        if Sw_t.device != torch.device("cpu"):
            Sw_t += torch.eye(Sw_t.shape[0]).to(Sw_t.get_device()) * self.r
        else:
            Sw_t += torch.eye(Sw_t.shape[0]) * self.r

        eig_prob_mat = torch.matmul(torch.linalg.inv(Sw_t), Sb_t)

        evals_t = torch.linalg.eigvals(eig_prob_mat)

        evals_t, _ = torch.sort(evals_t.real)  # ascending

        top_n_comp_evals = evals_t[
            -self.n_components :
        ]  # get top n_components (in general case, n_Comp = C-1)

        thresh = torch.min(top_n_comp_evals) + self.epsilon  # epsilon = 1

        top_k_evals = top_n_comp_evals[top_n_comp_evals <= thresh]
        if debug:
            print(
                "k: ",
                top_k_evals.shape[0],
                np.array_str(
                    evals_t.detach().cpu().numpy(), precision=3, suppress_small=True
                ),
            )
        loss = -torch.mean(top_k_evals)

        return loss


class Weighted_deepLDA(nn.Module):
    def __init__(
        self,
        num_classes,
        r=1e-3,
        n_components: int = None,
        epsilon: float = 1.0,
        normalize_grad=False,
    ):
        super().__init__()
        self.name = self.__class__.__name__
        self.loss_type = "dorfer"  # ؟‌
        self.num_classes = num_classes

        self.epsilon = epsilon
        self.r = r
        if n_components is None:
            self.n_components = num_classes - 1
        else:
            self.n_components = n_components

        self.normalize_grad = normalize_grad

    def forward(self, features, labels, confusion_mat: torch.Tensor, debug=False):
        """
        m = batch_size
        label - [m, ]
        feat - [m, dim] - in case of mnist toy example: [m, 2]
        """
        # TODO normalizing grads when facing test set (which doesnt require grad)
        if self.normalize_grad and features.requires_grad:
            features.register_hook(lambda grad: grad / grad.abs().max())

        mod_conf_mat = get_modified_conf_mat(confusion_mat)

        # sum over rows
        weights = (mod_conf_mat.sum(1) / (mod_conf_mat.sum(1).sum() + 1e-12)).to(
            features.device
        )

        print(
            "weights: ",
            np.array_str(weights.cpu().numpy(), precision=4, suppress_small=True),
        )

        #### St
        # cov_t = torch.cov(features.T)  # [d, d]

        #### Sw
        covs_w = []  # [c, d, d]
        for c in range(self.num_classes):
            curr_class_data = features[torch.where(labels == c)]
            temp = torch.cov(curr_class_data.T)
            covs_w.append(temp)
        covs_w = torch.stack(covs_w)
        # apply weights
        covs_w *= weights.view(weights.shape[0], 1, 1)
        Sw_t = torch.sum(covs_w, dim=0)  # from (c, d, d) to (d, d)

        #### Sb
        mu_t = torch.mean(features, axis=0).unsqueeze(1)  # [d, 1]
        mu_c = [
            torch.nan_to_num(torch.mean(features[torch.where(labels == i)], axis=0))
            for i in range(self.num_classes)
        ]  # [c, d]
        sbs = []
        for c in range(len(mu_c)):
            sbs.append(
                torch.mm(mu_c[c].unsqueeze(1) - mu_t, (mu_c[c].unsqueeze(1) - mu_t).T)
            )
        sbs = torch.stack(sbs)  # (c, d, d)
        # apply weights
        sbs *= weights.view(weights.shape[0], 1, 1)
        Sb_t = torch.sum(sbs, dim=0)  # from (c, d, d) to (d, d)

        # dorfer doesn't compute Sb directly:
        # Sb_t = St_t - Sw_t

        Sw_t += torch.eye(Sw_t.shape[0]).to(Sw_t.device) * self.r

        eig_prob_mat = torch.matmul(torch.linalg.inv(Sw_t), Sb_t)

        evals_t = torch.linalg.eigvals(eig_prob_mat)

        evals_t, _ = torch.sort(evals_t.real)
        top_n_comp_evals = evals_t[
            -self.n_components :
        ]  # get top n_components (in general case, n_Comp = C-1)

        thresh = torch.min(top_n_comp_evals) + self.epsilon  # epsilon = 1

        top_k_evals = top_n_comp_evals[top_n_comp_evals <= thresh]
        if debug:
            print(
                "k: ",
                top_k_evals.shape[0],
                np.array_str(
                    evals_t.detach().cpu().numpy(), precision=3, suppress_small=True
                ),
            )

        loss = -torch.mean(top_k_evals)

        return loss


class trace_st_trace_sb_Loss(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        """
        trace_st_trace_sb returns the following loss:

                tr(cov(h))                                      trace(st)
        loss = _________________ (for every c \in num_classes) = _____________
                tr(cov(E[h|c]))                                    trace(sb)

        where h is the representation of a training sample: c is the corresponding class
        (E[h|c]: mean of class 'c')

        Note: weighted Sw and Sb is used in this loss is weighted (get_scatters(weighted=True) by default)
        """
        super().forward(features, labels)

        st_list, sw_list, sb_list = get_scatters(features, labels, self.num_classes)
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]
        st = st_list.sum(dim=0)  # sum over dimensions , [1, ]
        sb = sb_list.sum(dim=0)  # sum over dimensions [1, ]

        # unused in loss, only report sw:
        sw = sw_list.sum()  # sum over dimensions and classes [1, ]

        loss = st / sb

        return loss


class trace_sw_trace_sb_Loss(liliBaseClass):

    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        """
        trace_sw_trace_sb returns the following loss:

                Σ E[tr(cov(h|c))]                                   trace(sw's)
        loss = ______________________ (for every c \in num_classes) = _____________
                tr(cov(E[h|c]))                                     trace(sb)

        where h is the representation of a training sample: c is the corresponding class
        (E[h|c]: mean of class 'c')

        Note: weighted Sw and Sb is used in this loss is weighted (get_scatters(weighted=True) by default)
        """
        super().forward(features, labels)

        st_list, sw_list, sb_list = get_scatters(features, labels, self.num_classes)
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        sw = sw_list.sum()  # sum over dimensions and classes [1, ]
        sb = sb_list.sum(dim=0)  # sum over dimensions [1, ]

        # unused in loss, only report st:
        st = st_list.sum(dim=0)  # sum over dimensions , [1, ]

        loss = sw / sb

        return loss


class max_st_trace_sb_Loss(liliBaseClass):

    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        """
        max_st_trace_sb returns the following loss:

                max_diagonal(cov(h))                                     max(st)       maximum variance along all dimensions
        loss = _______________________ (for every c \in num_classes) = _____________ = _______________________________________
                    tr(cov(E[h|c]))                                      trace(sb)

        where h is the representation of a training sample: c is the corresponding class
        (E[h|c]: mean of class 'c'), max_diagonal returns maximum diagonal element of matrix

        Note: weighted Sw and Sb is used in this loss is weighted (get_scatters(weighted=True) by default)
        """
        super().forward(features, labels)

        st_list, sw_list, sb_list = get_scatters(features, labels, self.num_classes)
        # st [d, ]
        # sw_list [c, d]
        # sb_list [d, ]

        st_max, max_idx = st_list.max(dim=0)  # max over dimensions , [1, ]
        sb = sb_list.sum()  # sum over dimensions [1, ]

        # unused in loss, only report sw:
        sw = sw_list.sum()  # sum over dimensions and classes [1, ]

        loss = st_max / sb

        return loss


from advertorch.context import ctx_noparamgrad_and_eval


class adversarial_wrapper(nn.Module):
    def __init__(self, model, loss_fn):
        super(adversarial_wrapper, self).__init__()
        if (isinstance(loss_fn, CL_Base) and not isinstance(loss_fn, CenterLoss)):
            # CenterLoss does not return nFeatures
            self.model = CCL_SingleOutputWrapper(model, loss_fn)
        else:
            self.model = SingleOutputWrapper(model)
        self.grad_layer = model.grad_layer

    def forward(self, x):
        return self.model(x)


class SingleOutputWrapper(nn.Module):
    def __init__(self, model):
        super(SingleOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        _, output2 = self.model(x)
        return output2


class CCL_SingleOutputWrapper(nn.Module):
    def __init__(self, model, loss_fn_object):
        super(CCL_SingleOutputWrapper, self).__init__()
        self.model = model
        self.loss_fn_object = loss_fn_object

    def forward(self, x):
        feats, _ = self.model(x)
        _, logits, _ = self.loss_fn_object(feats)
        return logits


class CCL(CL_not_weights):
    def __init__(self, num_classes: int, feat_dim: int, radius: float):
        super().__init__(num_classes, feat_dim)
        self.name = self.__class__.__name__
        self.loss_type = "ccl"  # ؟‌
        self.loss_fn = gring_loss(radius=radius)

        self.weights = nn.Parameter(
            torch.randn(self.num_classes, self.feat_dim)
        )  # [10, d]
        nn.init.xavier_uniform_(self.weights)

        self.centers = nn.Parameter(
            torch.zeros(self.num_classes, self.feat_dim, requires_grad=False),
            requires_grad=False,
        )

    def update_centers(self, model, loader, device, adversary=None, debug=False):
        self.centers = nn.Parameter(
            torch.zeros(
                self.num_classes, self.feat_dim, requires_grad=False, device=device
            ),
            requires_grad=False,
        )

        if not adversary is None:
            adv_model = adversarial_wrapper(model, self)
            print("updating adversarial centers...")
        else:
            print("updating clean centers...")

        for im, label in loader:
            im = im.to(device)
            label = label.to(device)

            if not adversary is None:
                with ctx_noparamgrad_and_eval(adv_model):
                    im = adversary.perturb(im, label)

            with torch.no_grad():
                feats, _ = model(im)

            for y_i in range(self.num_classes):
                self.centers[y_i] += torch.sum(feats[label == y_i], dim=0)

        with torch.no_grad():
            self.centers = nn.Parameter(
                self.centers.renorm(2, 0, 1e-5).mul(1e5).data, requires_grad=False
            )

    def forward(self, features, labels=None, debug=False):
        nFeatures = features.renorm(2, 0, 1e-5).mul(
            1e5
        )  # p = 2, dim = 0, maxnorm = 1e-5
        nWeights = self.weights.renorm(2, 0, 1e-5).mul(1e5)
        logits = torch.matmul(features, torch.transpose(nWeights, 0, 1))
        # Forward pass completed

        # gring
        if labels is None:
            gring_loss = None  # this should only happen when you want to only get the logits of the whole model
        else:
            gring_loss = self.loss_fn(features, self.centers[labels])

        return gring_loss, logits, nFeatures

class SCCL_Base(CL_weights, ABC):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__(num_classes, feat_dim)
        self.name = self.__class__.__name__
        self.loss_type = "SCCL_Base" 

        # update centers in warmup
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        nn.init.xavier_uniform_(self.centers)
    
    @abstractmethod
    def forward(self):
        pass

class SCCL(SCCL_Base):
    def __init__(self, num_classes: int, feat_dim: int, radius: float):
        super().__init__(num_classes, feat_dim)
        self.name = self.__class__.__name__
        self.loss_type = "sccl"
        self.loss_fn = gring_loss(radius=radius)

    def update_centers(self, model, loader, device, adversary=None, debug=False):
        self.centers = nn.Parameter(
            torch.zeros(
                self.num_classes, self.feat_dim, requires_grad=False, device=device
            ),
            requires_grad=False,
        )

        if not adversary is None:
            adv_model = adversarial_wrapper(model, self)
            print("updating adversarial centers...")
        else:
            print("updating clean centers...")

        for im, label in loader:
            im = im.to(device)
            label = label.to(device)

            if not adversary is None:
                with ctx_noparamgrad_and_eval(adv_model):
                    im = adversary.perturb(im, label)

            with torch.no_grad():
                feats, _ = model(im)

            for y_i in range(self.num_classes):
                self.centers[y_i] += torch.sum(feats[label == y_i], dim=0)

        with torch.no_grad():
            self.centers = nn.Parameter(
                self.centers.renorm(2, 0, 1e-5).mul(1e5).data, requires_grad=False
            )

    def forward(self, features, labels=None, debug=False):
        nFeatures = features.renorm(2, 0, 1e-5).mul(
            1e5
        )  # p = 2, dim = 0, maxnorm = 1e-5
        logits = torch.matmul(features, torch.transpose(self.centers, 0, 1))
        # Forward pass completed
        # print(self.centers[0][0], self.centers[0][0].grad)

        # gring
        if labels is None:
            gring_loss = None  # this should only happen when you want to only get the logits of the whole model
        else:
            gring_loss = self.loss_fn(features, self.centers[labels])

        return gring_loss, logits, nFeatures


class Cosine_SCCL(SCCL_Base):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__(num_classes, feat_dim)
        self.name = self.__class__.__name__
        self.loss_type = "cosine_sccl" 
       
    def update_centers(self, model, loader, device, adversary=None, debug=False):
        self.centers = nn.Parameter(
            torch.zeros(
                self.num_classes, self.feat_dim, requires_grad=False, device=device
            ),
            requires_grad=False,
        )

        if not adversary is None:
            adv_model = adversarial_wrapper(model, self)
            print("updating adversarial centers...")
        else:
            print("updating clean centers...")

        for im, label in loader:
            im = im.to(device)
            label = label.to(device)

            if not adversary is None:
                with ctx_noparamgrad_and_eval(adv_model):
                    im = adversary.perturb(im, label)

            with torch.no_grad():
                feats, _ = model(im)

            for y_i in range(self.num_classes):
                self.centers[y_i] += torch.sum(feats[label == y_i], dim=0)

        with torch.no_grad():
            self.centers = nn.Parameter(
                self.centers.renorm(2, 0, 1e-5).mul(1e5).data, requires_grad=False
            )

    def forward(self, features, labels=None, debug=False):
        nFeatures = features.renorm(2, 0, 1e-5).mul(
            1e5
        )  # p = 2, dim = 0, maxnorm = 1e-5
        logits = torch.matmul(features, torch.transpose(self.centers, 0, 1))
        if labels == None:
            cosine_loss = None
        else:
            cosine_similarity = torch.nn.functional.cosine_similarity(
                features, self.centers[labels]
            )
            cosine_distannce = 1 - cosine_similarity
            cosine_loss = torch.mean(cosine_distannce)
        return cosine_loss, logits, nFeatures


# Inherit from Function
class Baad(Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight, bias):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class MulConstant(Function):
    @staticmethod
    def forward(tensor, constant):
        return tensor * constant

    @staticmethod
    def setup_context(ctx, inputs, output):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor, constant = inputs
        ctx.constant = constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output, None  # grad_output * ctx.constant, None


class gring_loss(nn.Module):
    def __init__(self, radius: float, cut_grad: bool = False):
        super().__init__()
        self.name = self.__class__.__name__
        self.loss_type = "gring"  # ؟‌

        self.loss = nn.MSELoss()
        self.radius = radius
        self.cut_grad = cut_grad

    def forward(self, features, centers, debug=False):
        # centers_baad = self.radius * centers
        # aux_loss = 0.5 * self.loss(features, centers)

        # centers.retain_grad()
        # # centers_baad.retain_grad()
        # aux_loss.backward()

        # print('centers in forward\n', centers.grad)
        # # print('centers_baad\n', centers_baad.grad)

        # return aux_loss
        if self.cut_grad:
            aux_loss = 0.5 * self.loss(
                features, MulConstant.apply(centers, self.radius)
            )
        else:
            aux_loss = 0.5 * self.loss(features, centers * self.radius)
        return aux_loss

        
class CL_v0(CL_weights):
    """update Centers only based on CE loss and GringLoss"""

    def __init__(
        self, num_classes: int, feat_dim: int, radius: float, cut_grad: bool = False
    ):
        super().__init__(num_classes, feat_dim)
        self.name = self.__class__.__name__
        self.loss_type = "cl_v1"  # ؟‌
        self.cut_grad = cut_grad
        self.loss_fn = gring_loss(radius=radius, cut_grad=self.cut_grad)
        # self.loss_fn = gring_loss(radius=radius, cut_grad=True)

        # update centers in warmup
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features, labels=None, debug=False, grad=True):
        # nCenters = self.centers.renorm(2,0,1e-5).mul(1e5)
        # logits = torch.matmul(features, torch.transpose(nCenters, 0, 1))

        if grad:
            logits = torch.matmul(features, torch.transpose(self.centers, 0, 1))
        else:
            logits = torch.matmul(
                features, torch.transpose(self.centers.clone().detach(), 0, 1)
            )

        # gring
        if labels is None:
            gring_loss = None  # this should only happen when you want to only get the logits of the whole model
        else:
            if grad:
                gring_loss = self.loss_fn(features, self.centers[labels])
            else:
                gring_loss = self.loss_fn(
                    features, self.centers[labels].clone().detach()
                )

        return gring_loss, logits, None


class CL_v1(CL_weights):
    """update Centers only based on CE loss and GringLoss"""

    def __init__(
        self, num_classes: int, feat_dim: int, radius: float, cut_grad: bool = False
    ):
        super().__init__(num_classes, feat_dim)
        self.name = self.__class__.__name__
        self.loss_type = "cl_v2" 
        self.cut_grad = cut_grad
        self.loss_fn = gring_loss(radius=radius, cut_grad=self.cut_grad)

        # update centers in warmup
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features, labels=None, debug=False):
        # nCenters = self.centers.renorm(2,0,1e-5).mul(1e5)
        # logits = torch.matmul(features, torch.transpose(nCenters, 0, 1))

        logits = torch.matmul(features, torch.transpose(self.centers, 0, 1))

        # gring
        if labels is None:
            gring_loss = None  # this should only happen when you want to only get the logits of the whole model
        else:
            centers = self.centers[labels].clone().detach().requires_grad_(True)
            gring_loss = self.loss_fn(features, centers)

        return gring_loss, logits, None


class CL_v2(CL_weights):
    """update Centers only based on CE loss and GringLoss"""

    def __init__(
        self, num_classes: int, feat_dim: int, radius: float, cut_grad: bool = False
    ):
        super().__init__(num_classes, feat_dim)
        self.name = self.__class__.__name__
        self.loss_type = "cl_v3"  # ؟‌
        self.cut_grad = cut_grad
        self.loss_fn = gring_loss(radius=radius, cut_grad=self.cut_grad)

        # update centers in warmup
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features, labels=None, debug=False):
        # nCenters = self.centers.renorm(2,0,1e-5).mul(1e5)
        # logits = torch.matmul(features, torch.transpose(nCenters, 0, 1))

        logits = torch.matmul(features, torch.transpose(self.centers, 0, 1))

        # gring
        if labels is None:
            loss = None  # this should only happen when you want to only get the logits of the whole model
        else:
            # _, _, sb_list = get_scatters(
            #     features, labels, self.num_classes, weighted=True
            # )
            cosine_dist_centroids = cos_dist_centroids(
                features, labels, self.num_classes
            ).fill_diagonal_(float("Inf"))
            i, j = (
                cosine_dist_centroids == torch.min(cosine_dist_centroids)
            ).nonzero()[0]

            if debug:
                print(i, j)
            if i != j:
                min_dist = cosine_dist_centroids[i, j]
            # else:
            #     min_dist = sb_list.sum()

            gring_loss = self.loss_fn(features, self.centers[labels])

            loss = gring_loss / (min_dist + Constants.eta)

        return loss, logits, None


############################# In-Complete Ideas
class min_max_cov_mat_Loss(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        super().forward(features, labels)

        st_list, sw_list, sb_list = get_scatters(features, labels, self.num_classes)
        # st [d, ]
        # sw_list [c, d]
        # sb_list [c, d]

        cov_t, covs_w, cov_b = get_covs(features, labels, self.num_classes)
        # cov_t [d, d]
        # covs_t [c, d, d]
        # cov_b [d, d]

        # minimze the maximum variance in each class:
        sw_trace_max = torch.max(sw_list, dim=1).values.sum()
        sw_cov_ceffs = covs_w[:, 0, 1].abs().sum()
        # print(covs_w[:, 0, 1])

        loss = (sw_trace_max + sw_cov_ceffs) / sb_list.sum()

        return loss


class min_max_cov_mat_2_Loss(liliBaseClass):
    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        super().forward(features, labels)

        st_list, sw_list, sb_list = get_scatters(features, labels, self.num_classes)
        # st [d, ]
        # sw_list [c, d]
        # sb_list [c, d]

        cov_t, covs_w, cov_b = get_covs(features, labels, self.num_classes)
        # cov_t [d, d]
        # covs_t [c, d, d]
        # cov_b [d, d]

        centroids_pairwise_distance = get_dist_centroids(
            features, labels, self.num_classes
        )
        # [c, c]

        for curr_c in range(self.num_classes):
            # Numrator
            sw_max_curr_c = torch.max(sw_list[curr_c]).values

            # Denominator
            sb_min_curr_c = centroids_pairwise_distance[curr_c]

        # minimze the maximum variance in each class:
        sw_trace_max = torch.max(sw_list, dim=1).values.sum()
        sw_cov_ceffs = covs_w[:, 0, 1].abs().sum()
        # print(covs_w[:, 0, 1])

        loss = (sw_trace_max + sw_cov_ceffs) / sb_list.sum()

        return loss


# loss func based on cosine similarity:
def get_cosine_scatters(aux_outputs, labels, num_classes, weighted=True, debug=False):
    Ni = torch.bincount(
        labels, minlength=num_classes
    )  # size of each class in aux_outputs
    N = aux_outputs.shape[0]  # size of aux_outputs
    print(f"labels.shape: {labels.shape}")
    print(f"outputs.shape: {aux_outputs.shape}")

    overall_mean = torch.mean(
        aux_outputs, axis=0
    )  # [d, ], mean of all aux_outputs (separately over each dimension)
    mean_classes = (
        []
    )  # [c, d], mean_classes[i, j] -> within mean of class "i" over dimension "j"
    for c in range(num_classes):
        curr_class_mean = torch.nan_to_num(
            torch.mean(aux_outputs[torch.where(labels == c)], axis=0)
        )  # [d, ]
        mean_classes.append(curr_class_mean)

    """ St:
    # variance formula (over population):
    #            Σ (x_i - μ)
    # var(x) = ______________ ,
    #                n
    # x_i: the value of the one observation
    # μ: the mean value of all observations
    # n: the number of observations
    """

    def cosine_dist(u, v):
        return 1 - (torch.dot(u, v) / torch.norm(u, p=2) * torch.norm(v, p=2))

    st_list = (1 / N) * (
        (cosine_dist(aux_outputs, overall_mean))
        * (cosine_dist(aux_outputs, overall_mean))
    ).sum(
        dim=0
    )  # [d, ]

    """ Sb: 
    ^?
    """
    mean_dist_classes = (
        []
    )  # [c, d] , arr[class c]: weighted distance of mean of class c with overall mean
    for curr_c in range(num_classes):
        # cosine_dist(mean_classes[curr_c], overall_mean)
        temp = (cosine_dist(mean_classes[curr_c], overall_mean)) * (
            cosine_dist(mean_classes[curr_c], overall_mean)
        )  # [d, ]

        # multiply each mean_class distance by its relative frequency (Ni/N)
        if weighted:
            mean_dist_classes.append((Ni[curr_c] / N) * temp)
        else:
            mean_dist_classes.append(temp)

    mean_dist_classes = torch.stack(mean_dist_classes, dim=0)
    if weighted:
        sb_list = mean_dist_classes.sum(dim=0)  # [d, ]
    else:
        sb_list = (1 / num_classes) * mean_dist_classes.sum(dim=0)  # [d, ]

    # NOT equivalent to :
    # temp2 = torch.var(torch.stack(mean_classes), dim=0, unbiased=False) # [d, ]
    # print(temp2, sb_list)

    """ Sw: 
    # calculate below formula wihtin each class
    #            Σ (x_i - μ)
    # var(x) = ______________ ,
    #                n
    ##### weighted ^?
    """
    sw_list = []  # [c, d]
    for curr_c in range(num_classes):

        curr_class_data = torch.nan_to_num(
            aux_outputs[torch.where(labels == curr_c)]
        )  # datas with label = curr_c

        if curr_class_data.shape[0] == 0:
            print(labels)
            print(f"current batch does not include class {curr_c}.")
            # exit()

        # cosine_dist(curr_class_data , mean_classes[curr_c])
        temp = (cosine_dist(curr_class_data, mean_classes[curr_c])) * (
            cosine_dist(curr_class_data, mean_classes[curr_c])
        )
        temp = (1 / Ni[curr_c]) * temp.sum(
            dim=0
        )  # [d, ] , mean over all curr_class_data

        # equivalent to:
        # temp2 = torch.var(curr_class_data, dim=0, unbiased=False) # [d, ]
        # print('sw', temp, temp2)

        # multiply each class by relative frequency (sw is weighted variances)
        if weighted:
            sw_list.append((Ni[curr_c] / N) * temp)
        else:
            sw_list.append(temp)

    sw_list = torch.stack(sw_list)

    # debug=True
    if debug:
        print("sw: {:.4f}".format(sw_list.sum().item()))
        # print(sw_list)
        print("sb: {:.4f}".format(sb_list.sum().item()))
        print("st: {:.4f}".format(st_list.sum().item()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return st_list.to(device), sw_list.to(device), sb_list.to(device)


class cosine_loss(nn.Module):
    def __init__(self, num_classes, feat_dim, r, n_components, normalize_grad=False):
        super().__init__()
        self.name = self.__class__.__name__
        self.loss_type = "dorfer"
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.r = r
        self.n_components = n_components  # ?
        self.normalize_grad = normalize_grad

    def forward(self, features, labels):
        """
        m = batch_size
        label - [m, ]
        feat - [m, dim] - in case of mnist toy example: [m, 2]
        """
        # dump = [features.cpu().detach().numpy(), labels.cpu().detach().numpy()]
        # with open('features_labels.pickle', 'wb') as handle:
        #     pickle.dump(dump, handle, protocol=1)

        # TODO normalizing grads when facing test set (which doesnt require grad)
        if self.normalize_grad and features.requires_grad:
            features.register_hook(lambda grad: grad / grad.abs().max())

        St_t, Sw_t, Sb_t = get_cosine_scatters(features, labels, self.num_classes)
        Sw_t = torch.mean(Sw_t, dim=0)  # from (c, d, d) to (d, d)

        # just like DORFER:
        Sb_t = St_t - Sw_t

        # try to calculate Sw t b just like what dorfer did:
        # https://github.com/CPJKU/deep_lda/blob/e47755bae54a5cb33f59c3954aaefd6f84b91cd0/deeplda/models/mnist_dlda.py#L119

        if Sw_t.device != torch.device("cpu"):
            Sw_t += torch.eye(Sw_t.shape[0]).to(Sw_t.get_device()) * self.r
        else:
            Sw_t += torch.eye(Sw_t.shape[0]) * self.r

        eig_prob_mat = torch.matmul(torch.linalg.inv(Sw_t), Sb_t)

        evals_t = torch.linalg.eigvals(eig_prob_mat)

        evals_t, _ = torch.sort(evals_t.real)
        top_n_comp_evals = evals_t[
            -self.n_components :
        ]  # get top n_components (in general case, n_Comp = C-1)

        thresh = torch.min(top_n_comp_evals) + 1.0  # epsilon = 1

        top_k_evals = top_n_comp_evals[top_n_comp_evals <= thresh]
        # print('k in dorfer loss: ', top_k_evals.shape[0])

        loss = -torch.mean(top_k_evals)

        return loss


class det_sb_det_sw_minus(liliBaseClass):
    """
    based on http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_LDA09.pdf, page 33
    objective function is to maximize det(Sb)/det(Sw)
    """

    def __init__(self, num_classes, normalize_grad=False) -> None:
        super().__init__(num_classes, normalize_grad)
        self.name = self.__class__.__name__

    def forward(self, features, labels):
        super().forward(features, labels)
        St, Sws, Sb = get_covs(features, labels, self.num_classes)
        Sw = Sws.sum(dim=0)

        loss = -(torch.linalg.det(Sb) / torch.linalg.det(Sw))

        return loss


class Trades(nn.Module):
    """https://github.com/yaodongyu/TRADES/blob/master/trades.py"""

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.loss_type = "Trades"

    def forward(self, adv_logits, cln_logits):
        criterion_kl = nn.KLDivLoss(size_average=False)

        batch_size = len(adv_logits)
        eps = 1e-30
        adv_out = F.log_softmax(adv_logits, dim=1) + eps
        cln_out = F.softmax(cln_logits, dim=1) + eps

        loss = (1.0 / batch_size) * criterion_kl(adv_out, cln_out)

        return loss
    
    
class Boosted_CE(nn.Module): 
    """
    Used in MART paper
    """
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.loss_type = "CE"

    def forward(self, logits, targets):
        
        probs = F.softmax(logits, dim=1)

        tmp1 = torch.argsort(probs, dim=1)[:, -2:] 
        # [bs, 2]
        # tmp1[i][0]: Second biggest probs class for i-th sample
        # tmp1[i][1]: The    biggest probs class for i-th sample

        new_y = torch.where(tmp1[:, -1] == targets, tmp1[:, -2], tmp1[:, -1])
        # [bs, 1] or [bs]
        # new_y[i] or new_y[i][0]: second biggest prob class for i-th sample (ONLY MISSCLASSIDIEDS???)
        
        loss = F.cross_entropy(logits, targets) + \
                F.nll_loss(torch.log(1.0001 - probs + 1e-12), new_y)

        return loss

class MART(nn.Module):
    """
    From the paper `https://openreview.net/forum?id=rklOg6EFwS`
    Code from https://github.com/YisenWang/MART 
    This loss_Function just returns the second term in the MART loss function
    """
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.loss_type = "MART"

    def forward(self, adv_logits, cln_logits, adv_targets, cln_targets):
        batch_size = cln_logits.shape[0]
        
        adv_probs = F.softmax(adv_logits, dim=1)
        cln_probs = F.softmax(cln_logits, dim=1)

        true_probs = torch.gather(cln_probs, 1, (cln_targets.unsqueeze(1)).long()).squeeze()

        kl = nn.KLDivLoss(reduction='none')
        loss_robust = (1.0 / batch_size) * torch.sum(
            torch.sum( kl(torch.log(adv_probs + 1e-12), cln_probs) , dim=1) * (1.0000001 - true_probs)
        )

        return loss_robust


class ALP(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.loss_type = "ALP" 
        self.loss_fn = nn.MSELoss()

    def forward(self, adv_logits, cln_logits):
        loss = 0.5 * self.loss_fn(adv_logits, cln_logits).mean()
        return loss

if __name__ == "__main__":
    # Sample features:
    # assuming f-space is 512 dimenstional, and there are 10 classes
    # and batch size is 128
    torch.manual_seed(11)  # for reproducibility
    num_classes = 10
    bs = 128
    feat_dim = 512
    feats = torch.randn(bs, feat_dim)
    labels = torch.randint(0, num_classes, (bs,))

    print(feats.shape)
    print(labels.shape)
    print("class frequency in the sample batch: ", labels.unique(return_counts=True))

    # Testing j1:
    loss_fn_j1 = j1_Loss(
        num_classes=num_classes, normalize_grad=False
    )  # ignore `normalize_grad` for now
    batch_loss = loss_fn_j1(feats, labels)
    print(batch_loss)

    # Testing Lili's implementation of j1:
    loss_fn_lili = vect_mc_discriminant_Loss(
        num_classes=num_classes, normalize_grad=False
    )  # ignore `normalize_grad` for now
    batch_loss = loss_fn_lili(feats, labels)
    print(batch_loss)

    # Testing proposed method:
    loss_fn_prop = w_j1_Loss(
        num_classes=num_classes, normalize_grad=False
    )  # ignore `normalize_grad` for now
    # A dummy confusion matrix:
    import sklearn
    from sklearn.metrics import confusion_matrix

    dummy_conf = torch.tensor(
        confusion_matrix(torch.randint(0, num_classes, (bs,)).numpy(), labels.numpy())
    )
    batch_loss = loss_fn_prop(feats, labels, dummy_conf)
    print(batch_loss)


