import torch
import torch.nn as nn
import numpy as np
import time, os, random, copy
import os
import pandas as pd
from math import pi, cos, sin
import matplotlib.pyplot as plt
import inspect
import imageio
from typing import Dict, List
import re
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

# relative import hacks (sorry)
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) # for bash user
os.chdir(parentdir) # for pycharm user
from utils.custom_data import *
from utils.loss_function_utils import get_modified_conf_mat

def make_dir(path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)


import matplotlib.pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

if __name__ == '__main__':
    # feats = np.random.randn(100, 2)
    feats = np.array([
        [-2, 2],
        [3, 4],
        [0, 0],
        [1, 1]
    ])
    print(feats.shape)
    norms = np.sqrt(np.power(feats, 2).sum(1))
    print(norms)

import plotly.express as px
import plotly.graph_objects as go

def get_color_temp(id: int):
    c = ['#ffe119', '#a9a9a9', '#4363d8', '#f58231', '#ffff00',
         '#ff00ff', '#009900', '#999900', '#00ff00', 'black']
    return c[id]

def plotly_2d_html_save(feats, targets, runPath: str = None, name: str = None):

    fig = go.Figure()
    for class_num in range(len(np.unique(targets))): 
        feats_c = feats[targets==class_num]
        fig.add_trace(
            go.Scatter(
                x=feats_c[:, 0], y=feats_c[:, 1],
                mode='markers',
                marker_color = get_color_temp(class_num),
                marker_size = 12,
                opacity=0.8
            )
        )
    make_dir(runPath)
    fig.write_html(f'{runPath}/{name}.html')

"""
set coeff: 
feats_norm = np.sqrt(np.power(feats, 2).sum(1))
feats_max_norm = np.max(feats_norm)
weights_norm = np.sqrt(np.power(weights, 2).sum(1))
weights_max_norm = np.max(weights_norm)
coeff = feats_max_norm / weights_max_norm
"""
import math
import scipy

def get_angle_origin(vec): 
    
    vectors = np.array([vec, [1, 0]])
 
    dist = scipy.spatial.distance.pdist(vectors, 'cosine')
 
    angle = np.rad2deg(np.arccos(1 - dist))
    coss = 1 - dist
    # print(coss)
    res = angle[0] 
    if vec[1] < 0: 
        res = 360 - res

    return res, coss[0]

def get_boundaries(w, line_length):

    angle_arr = []
    cos_arr = []

    # print(w)
    # print(w.shape)

    for i in range(10):
        w_i = w[i, :]
        angle, cos = get_angle_origin(w_i)
        angle_arr.append(angle)
        cos_arr.append(cos)

    angle_arr = np.array(angle_arr)
    cos_arr = np.array(cos_arr)

    angl_perm = np.argsort(angle_arr)
    # print(angl_perm)

    ## end of globals 

    def temp2(w_1_idx, w_2_idx, alpha): 
        # print('--', w_1_idx, w_2_idx)
        angle1 = angle_arr[w_1_idx]
        angle2 = angle_arr[w_2_idx]
        # print('angles...', angle1, angle2)

        # print('alpha', alpha)
        is_class1 = []
        angle_label = []
        for counter in np.linspace(0, 360, 3601):
            t = angle1 + counter % 360
            theta1 = (t - angle1) % 360
            theta2 = (t - angle2) % 360 
            
            score1 = np.linalg.norm(w[w_1_idx, :]) * np.cos(np.deg2rad(theta1))
            score2 = np.linalg.norm(w[w_2_idx, :]) * np.cos(np.deg2rad(theta2))
            # print(f'{theta1: .3f} vs {(theta2): .3f}', f'score1: {score1:.3f}', f'- score2: {score2:.3f}')
            
            is_class1.append([np.cos(np.deg2rad(theta1+angle_arr[w_1_idx])), np.sin(np.deg2rad(theta1+angle_arr[w_1_idx])), float(score1 >= score2)])
            angle_label.append([theta1+angle_arr[w_1_idx], float(score1 >= score2)])

        angle_label = np.array(angle_label)

        angle_label = angle_label % 360
        angle_label = angle_label[np.argsort(angle_label[:, 0], axis=0)]
        
        # print(angle_label)
        # exit()
        return angle_label, is_class1

    def get_st_end(R): 
        mask = R[:, 1] == 1.0
        start_indices = np.where(mask & ~np.roll(mask, 1))[0]
        end_indices = np.where(mask & ~np.roll(mask, -1))[0]
        if mask[0] and mask[-1]:
            end_indices[0] = end_indices[-1]

        # print("Indices where consecutive 1's start:", start_indices)
        # print("Indices where consecutive 1's end:", end_indices)
        return start_indices[0], end_indices[0]

    def temp(w_idx, line_length, ax = None):

        w_angle_idx = np.where(angl_perm==w_idx)[0][0]  
        prev_idx = angl_perm[(w_angle_idx-1)%10]
        next_idx = angl_perm[(w_angle_idx+1)%10]

        # print(f'comparing class {w_idx} with next({next_idx}) and prev({prev_idx})')
        # w_i    = w[w_idx]
        # w_next = w[next_idx]
        # w_last = w[prev_idx]

        angle_from_next = (angle_arr[next_idx]-angle_arr[w_idx]) % 360
        angle_from_prev = (angle_arr[w_idx]   -angle_arr[prev_idx]) % 360

        angle_label_prev, points1 = temp2(w_idx, prev_idx, angle_from_prev)
        angle_label_next, points2 = temp2(w_idx, next_idx, angle_from_next)
        
        # print(angle_label_prev)
        # print(angle_label_next)

        points1 = np.array(points1)
        points2 = np.array(points2)

        intersect_angles = angle_label_prev
        intersect_angles[:, 1] = intersect_angles[:, 1] * angle_label_next[:, 1]
        st, end = get_st_end(intersect_angles)
        # intersect_angles = intersect_angles[intersect_angles[:, 1] == 1.0]
        
        angle1 = intersect_angles[st][0]
        angle2 = intersect_angles[end][0]
        
        # print('ang1, ang2:', angle1, angle2)
        
        x1 = line_length * np.cos(np.deg2rad(angle1))
        y1 = line_length * np.sin(np.deg2rad(angle1))
        x2 = line_length * np.cos(np.deg2rad(angle2))
        y2 = line_length * np.sin(np.deg2rad(angle2))

        arc1 = points1[points1[:, 2]==1.0] * 1
        arc2 = points2[points2[:, 2]==1.0] * 1.2
        
        intersect_arc = np.copy(points1)
        intersect_arc[:, 2] = intersect_arc[:, 2] * points2[:, 2]
        intersect_arc = intersect_arc[intersect_arc[:, 2] == 1.]

        if not ax is None: 
            
            ax.scatter(arc1[:, 0], arc1[:, 1], c = arc1[:, 2])
            ax.scatter(arc2[:, 0], arc2[:, 1], c = arc2[:, 2]) 
            # ax.scatter(intersect_arc[:, 0], intersect_arc[:, 1])

            for i in range(10):
                ax.plot([0, w[i, 0]], [0, w[i, 1]])
                ax.annotate(f'{i}: {angle_arr[i]:.2f} | size: {np.linalg.norm(w[i, :]):.3f}', [w[i, 0], w[i, 1]])
        
            ax.axis('square')

        return x1, y1, x2, y2

    locs = []
    for i in range(10):
        locs.append(temp(i, line_length=line_length))
    
    return locs

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

def angles_between_samples_and_vector(F, W):
    norm_F = np.linalg.norm(F, axis=1, keepdims=True)
    norm_W = np.linalg.norm(W)
    cosine_similarity = np.dot(F, W) / (norm_F * norm_W)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    angles = np.arccos(cosine_similarity)
    return np.diag(np.rad2deg(angles))


def count_vectors_in_orthants_vectorized(F):
    orthant_indices = np.sum((F < 0) * 2**np.arange(F.shape[1]), axis=1)
    counts = np.bincount(orthant_indices, minlength=2**F.shape[1])
    return orthant_indices, counts

def _mean_within_mse(feats, labels, weights): 
    mse = nn.MSELoss(reduction='none')
    
    
    gloabl_mean_mse = []
    for class_num in range(len(np.unique(labels))):
        feats_c = feats[labels==class_num] # [k, 2]
        mse_means = mse(torch.tensor(feats_c), torch.tensor(weights[labels[labels==class_num]])).numpy().mean(1)
        gloabl_mean_mse.append(mse_means.mean())

    return np.array(gloabl_mean_mse).mean()


def mse_hist(feats, labels, weights, path_to_save: str, name_to_save: str): 
    mse = nn.MSELoss(reduction='none')
    
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 8), sharex=True) # , sharey=True, sharex=True
    axes = axes.flatten()
    for i in range(len(axes)):
        if i >= 10:
            fig.delaxes(axes[i])

    gloabl_mean_mse = []
    for class_num in range(len(np.unique(labels))):
        feats_c = feats[labels==class_num] # [k, 2]
        mse_means = mse(torch.tensor(feats_c), torch.tensor(weights[labels[labels==class_num]])).numpy().mean(1)
        gloabl_mean_mse.append(mse_means.mean())

        axes[class_num].hist(mse_means, bins=100)
        axes[class_num].set_title(f'{class_num}') 
        axes[class_num].set_xlabel('mean MSE')
        axes[class_num].set_ylabel('freq')    
        axes[class_num].grid(True)

    gloabl_mean_mse = np.array(gloabl_mean_mse).mean()
    plt.tight_layout()
    plt.legend()
    make_dir(path_to_save)
    plt.suptitle(f'gloabl_mean_mse: {gloabl_mean_mse}, Hist Mean of mse across dimensions')
    plt.savefig(f'{path_to_save}/{name_to_save}.jpg')
    plt.clf()

def _angle_analyis(features: torch.tensor, labels: torch.tensor, preds: torch.tensor, weights: torch.tensor):
    mean_classes = []
    std_classes = []
    for class_num in range(len(torch.unique(labels))):
        feats_c = features[labels==class_num] # [k, d]
        std_classes.append(torch.std(feats_c, dim=0)) # [d, ]
        mean_c = torch.mean(feats_c, dim=0) # [d, ]
        mean_classes.append(mean_c)
    mean_classes = torch.vstack(mean_classes) # [C, d]
    std_classes = torch.vstack(std_classes) # [C, d]

    mean_within_std = torch.mean(std_classes) # [1, ]
    std_within_std = torch.std(std_classes)
    mean_std_across_d_of_within_std = torch.mean(torch.std(std_classes, dim=1)) # mean([C, ]) = [1, ] 
    mean_std_across_C_of_within_std = torch.mean(torch.std(std_classes, dim=0)) # mean([d, ]) = [1, ] 
    
    max_within_std = torch.max(std_classes)
    min_within_std = torch.min(std_classes)
    
    mean_between_var = []
    var_between_var = []
    max_between_var = []
    min_between_var = []
    
    min_cos_dist = []
    max_cos_dit = []
    mean_cos_dist = []




    w_mu_angle_mat, _ = get_degree_cosine(weights, mean_classes)
    w_w_angle_mat, _ = get_degree_cosine(weights, weights, zero_diag=True)
    mu_mu_angle_mat, _ = get_degree_cosine(mean_classes, mean_classes, zero_diag=True)

    mean_within_std_angle_feats_W = []
    for class_num in range(len(np.unique(labels))):
        feats_c = features[labels==class_num] # [k, 2]
        angles = angles_between_samples_and_vector(feats_c, weights[class_num])
        mean_within_std_angle_feats_W.append(angles.std())

    mean_within_std_angle_feats_W = np.mean(np.array(mean_within_std_angle_feats_W))


def geometric_analysis(
        features: np.ndarray, 
        labels: np.ndarray,
        preds: np.ndarray, 
        weights: np.ndarray,
        name_to_save: str, 
        path_to_save: str,
        title: str = None,
        zoom: bool = False
    ):
    output = {}

    mse_hist(feats=features, labels=labels, weights=weights, 
             name_to_save=f'{name_to_save} Hist MSE', path_to_save=path_to_save)
    
    # Create a 2x2 subplot grid with custom widths for columns
    fig = plt.figure(figsize=[20, 10], dpi=200)
    gs = gridspec.GridSpec(2, 3, width_ratios=[0.25, 0.25, 0.5])
    """
    [0, 0] | [0, 1] | [0, 2]
    [1, 0] | [1, 1] | [1, 2]
    """

    # Create a subplot that spans two rows (belongs to the first column)
    conf_mat_ax = plt.subplot(gs[0, 0])
    w_mu_angles_ax = plt.subplot(gs[1, 0])
    w_w_angles_ax = plt.subplot(gs[0, 1])
    mu_mu_angles_ax = plt.subplot(gs[1, 1])
    ax3 = plt.subplot(gs[:, 2])

    c = ['black', '#0000ff', '#990000', '#00ffff', '#ffff00',
        '#ff00ff', '#009900', '#999900', '#00ff00', '#009999']    
    
    # plot features, mean, weights, based on True label
    if features.shape[0] < 2000 or zoom: marker_size = 4
    elif features.shape[0] < 11000: marker_size = 3
    else: marker_size = 1
    
    # get boundaries
    feat_dim = features.shape[1]

    if feat_dim == 2: 
        feats_to_plot = features
        weights_to_plot = weights

        max_norm_f = np.quantile(np.linalg.norm(feats_to_plot, axis=1), q=0.95)    
        locs = get_boundaries(weights_to_plot, max_norm_f)
    else: 
        data = np.vstack([features, weights])
        pca = PCA(n_components=2)
        pca.fit(data)
        feats_to_plot = pca.transform(features)
        weights_to_plot = pca.transform(weights)
        
    ############################################################################################
    curr_ax = ax3
    mean_classes_plot = []
    max_norm = -1.0     
    w_max_norm = -1.0
    for class_num in range(len(np.unique(labels))):
        feats_c = feats_to_plot[labels==class_num] # [k, 2]
        mean_c = np.mean(feats_c, axis=0) # [2, ]
        mean_classes_plot.append(mean_c)
        max_norm = max(max_norm, np.linalg.norm(mean_c))
        w_max_norm = max(w_max_norm, np.linalg.norm(weights_to_plot[class_num, :]))

    ### r = 0.1 #### only when you project all features into a sphere
    for class_num in range(len(np.unique(labels))):
        feats_c = feats_to_plot[labels==class_num] ### * r # [k, 2]
        ### r += 0.1
        
        curr_ax.annotate(f'{class_num}', [mean_classes_plot[class_num][0], mean_classes_plot[class_num][1]])
        
        curr_ax.plot(feats_c[:, 0], 
                    feats_c[:, 1],
                    '.', c=c[class_num], alpha=0.5, 
                    ms=marker_size, 
                    label=class_num)

        if feat_dim == 2:
            # plot (resized) weights
            resized_weight = (weights_to_plot[class_num, :]/w_max_norm) * (max_norm)
            curr_ax.plot([0, resized_weight[0]], [0, resized_weight[1]],
                        c='black', lw=2, label=class_num)

            curr_ax.plot([0, resized_weight[0]], [0, resized_weight[1]],
                        c=c[class_num], lw=1.5, label=class_num)
            
            curr_ax.annotate(f'{class_num}', [resized_weight[0], resized_weight[1]])
            
            # plot boundaries
            x1, y1, x2, y2 = locs[class_num]

            plt.plot([0, x1], [0, y1], [0, x2], [0, y2], c=c[class_num], label=f'Both Angles', alpha=0.4, lw=1.25)
            plt.fill_between([0, x1, x2, 0], [0, y1, y2, 0], color=c[class_num], alpha=0.1)
        
    mean_classes_plot = np.vstack(mean_classes_plot)
    curr_ax.set_title(name_to_save)
    # ax.ylim(-200, 200)
    # ax.xlim(-200, 200)
    curr_ax.axis('square')
    if zoom: 
        curr_ax.set_xlim(2*min(mean_classes_plot[1, 0], mean_classes_plot[4, 0]), 2*max(mean_classes_plot[1, 0], mean_classes_plot[4, 0]))
        curr_ax.set_ylim(-2, 2*max(mean_classes_plot_classes[1, 1], mean_classes_plot[4, 1]))
    ############################################################################################

    mean_classes = []
    for class_num in range(len(np.unique(labels))):
        feats_c = features[labels==class_num] # [k, d]
        mean_c = np.mean(feats_c, axis=0) # [d, ]
        mean_classes.append(mean_c)
    mean_classes = np.vstack(mean_classes)

    # conf matrix 
    curr_ax = conf_mat_ax
    acc = accuracy_score(labels, preds)
    conf = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(conf)
    disp.plot(ax=curr_ax, colorbar=False)
    curr_ax.set_title(f'acc: {acc}')

    ## w_mu: 
    curr_ax = w_mu_angles_ax
    w_mu_angle_mat, _ = get_degree_cosine(weights, mean_classes)
    curr_ax.set_title('w_mu_angles')
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    im = curr_ax.imshow(np.zeros((10, 10)))
    curr_ax.set_xticks(np.arange(len(classes)))
    curr_ax.set_yticks(np.arange(len(classes)))
    plt.setp(curr_ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = curr_ax.text(j, i, f'{w_mu_angle_mat[i, j]: 0.2f}',
                        ha="center", va="center", color="w", fontdict={'size': 8})

    ## w_w: 
    curr_ax = w_w_angles_ax
    w_w_angle_mat, _ = get_degree_cosine(weights, weights, zero_diag=True)
    curr_ax.set_title('w_w_angles')
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    im = curr_ax.imshow(np.zeros((10, 10)))
    curr_ax.set_xticks(np.arange(len(classes)))
    curr_ax.set_yticks(np.arange(len(classes)))
    plt.setp(curr_ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = curr_ax.text(j, i, f'{w_w_angle_mat[i, j]: 0.2f}',
                        ha="center", va="center", color="w", fontdict={'size': 8})

    ## mu_mu: 
    curr_ax = mu_mu_angles_ax
    mu_mu_angle_mat, _ = get_degree_cosine(mean_classes, mean_classes, zero_diag=True)
    curr_ax.set_title('mu_mu_angles')
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    im = curr_ax.imshow(np.zeros((10, 10)))
    curr_ax.set_xticks(np.arange(len(classes)))
    curr_ax.set_yticks(np.arange(len(classes)))
    plt.setp(curr_ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    # print(angle_mat)
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = curr_ax.text(j, i, f'{mu_mu_angle_mat[i, j]: 0.2f}',
                        ha="center", va="center", color="w", fontdict={'size': 8})

    if title is None:
        plt.title(f'{name_to_save}')
    else:
        plt.title(title)
        
    plt.tight_layout()
    make_dir(path_to_save)
    plt.suptitle(f'{feat_dim}-D features')
    plt.savefig(f'{path_to_save}/{name_to_save}.jpg')
    plt.clf()

    ################## Angles histogram
    ### correctly classified features with Weights
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 8), sharex=True) # , sharey=True, sharex=True
    axes = axes.flatten()
    for i in range(len(axes)):
        if i >= 10:
            fig.delaxes(axes[i])

    mean_std_angles_W = []
    for class_num in range(len(np.unique(labels))):
        # feats_c = features[labels==class_num] # [k, 2]
        # angles = angles_between_samples_and_vector(feats_c, weights[class_num])
        # axes[class_num].hist(angles, bins=180, range=(0.0, 180.0))
        # axes[class_num].set_title(f'{class_num}') 

        correct_indicies = np.array(preds == class_num) & np.array(labels==class_num)
        temp2 = 0
        if np.sum(correct_indicies) > 0: 
            correct_feats_c = features[correct_indicies]
            correct_angles = angles_between_samples_and_vector(correct_feats_c, weights[class_num])
            mean_std_angles_W.append(correct_angles.std())
            axes[class_num].hist(correct_angles, bins=180, label="Correct", alpha=1.0, range=(0.0, 180.0)) # , range=(0.0, 180.0), density=True
            temp2 = np.mean(correct_angles)
        
        temp = angles_between_samples_and_vector(mean_classes[class_num][None, :], weights[class_num])
        axes[class_num].set_title(f'{class_num}') # , mean:{temp2}, w_mu: {temp} 
        axes[class_num].set_xlabel('angle')
        axes[class_num].set_ylabel('freq')    
        axes[class_num].grid(True)

    mean_std_angles_W = np.mean(np.array(mean_std_angles_W))
    output['mean_std_angles_W'] = mean_std_angles_W
    plt.tight_layout()
    plt.legend()
    make_dir(path_to_save)
    plt.suptitle(f'Mean_angle:{mean_std_angles_W},angle of correctly classified feats and W')
    plt.savefig(f'{path_to_save}/{name_to_save} correctly classified W histogram.jpg')
    plt.clf()

    ### misclassified features with Weights
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 8), sharex=True) # , sharey=True, sharex=True
    axes = axes.flatten()
    for i in range(len(axes)):
        if i >= 10:
            fig.delaxes(axes[i])

    for class_num in range(len(np.unique(labels))):   
        incorrect_indicies = np.array(preds != class_num) & np.array(labels==class_num)
        if np.sum(incorrect_indicies) > 0: 
            incorrect_feats_c = features[incorrect_indicies] # [k, 2]
            incorrect_angles = angles_between_samples_and_vector(incorrect_feats_c, weights[class_num])
            axes[class_num].hist(incorrect_angles, bins=180, label="Miss", alpha=1.0, range=(0.0, 180.0)) # , range=(0.0, 180.0), density=True
            
        axes[class_num].set_title(f'{class_num}') 
        axes[class_num].set_xlabel('angle')
        axes[class_num].set_ylabel('freq')    
        axes[class_num].grid(True)

    plt.tight_layout()
    plt.legend()
    make_dir(path_to_save)
    plt.suptitle('angle of misclassified features and class weight (degree)')
    plt.savefig(f'{path_to_save}/{name_to_save} misclassified W histogram.jpg')
    plt.clf()

    ################ MU HISTOGRAMS
    #### correct features with Centers
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for i in range(len(axes)):
        if i >= 10:
            fig.delaxes(axes[i])

    for class_num in range(len(np.unique(labels))):
        correct_indicies = np.array(preds == class_num) & np.array(labels==class_num)
        if np.sum(correct_indicies) > 0:     
            feats_c = features[correct_indicies] # [k, 2]
            angles = angles_between_samples_and_vector(feats_c, mean_classes[class_num])
            axes[class_num].hist(angles, bins=180, range=(0.0, 180.0))
            axes[class_num].set_title(f'{class_num}') 
    
    plt.tight_layout()
    make_dir(path_to_save)
    plt.suptitle('angle of Correctly classified features and class mean (degree)')
    plt.savefig(f'{path_to_save}/{name_to_save} correctly classified MU histogram.jpg')
    plt.clf()

    
    #### incorrect features with Centers
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for i in range(len(axes)):
        if i >= 10:
            fig.delaxes(axes[i])

    for class_num in range(len(np.unique(labels))):
        incorrect_indicies = np.array(preds != class_num) & np.array(labels==class_num)
        if np.sum(incorrect_indicies) > 0:     
            feats_c = features[incorrect_indicies] # [k, 2]
            angles = angles_between_samples_and_vector(feats_c, mean_classes[class_num])
            axes[class_num].hist(angles, bins=180, range=(0.0, 180.0))
            axes[class_num].set_title(f'{class_num}') 

    plt.tight_layout()
    make_dir(path_to_save)
    plt.suptitle('angle of misclassified features and class mean (degree)')
    plt.savefig(f'{path_to_save}/{name_to_save} misclassified MU histogram.jpg')
    plt.clf()

    ######################### Orthants Mat
    if feat_dim <= 5: 
        fig, ax = plt.subplots(dpi=120)
        
        orthants_count = []
        ws_orthant = []
        mus_orthant = []
        for class_num in range(len(np.unique(labels))):
            feats_c = features[labels==class_num] # 
            _, curr_class_orthants_count = count_vectors_in_orthants_vectorized(feats_c)
            w_orthant, _ = count_vectors_in_orthants_vectorized(weights[class_num][None, :])
            mu_orthant, _ = count_vectors_in_orthants_vectorized(mean_classes[class_num][None, :])
            
            orthants_count.append(curr_class_orthants_count)
            ws_orthant.append(w_orthant[0])
            mus_orthant.append(mu_orthant[0])

        orthants_count = np.vstack(orthants_count)
        orthants_count = orthants_count / np.sum(orthants_count, axis=1)[:, None] # normalize
        
        ax.set_title('orphants (r=Weight, b=Mu, y=both)')
        classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        im = ax.imshow(np.zeros((10, 2**feat_dim)))
        ax.set_xticks(np.arange(2**feat_dim))
        ax.set_yticks(np.arange(len(classes)))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        for i in range(len(classes)):
            for j in range(2**feat_dim):
                if j == ws_orthant[i] and j == mus_orthant[i]: 
                    color = 'y'
                elif j == ws_orthant[i]: 
                    color = 'r'
                elif j == mus_orthant[i]: 
                    color = 'b'
                else:   
                    color = 'w'
                        
                text = ax.text(j, i, f'{orthants_count[i, j]: 0.3f}',
                            ha="center", va="center", color=color, fontdict={'size': 8})
        
        plt.tight_layout()
        make_dir(path_to_save)
        plt.savefig(f'{path_to_save}/Orthants Matrix-{name_to_save}.jpg')
        plt.clf()

    return output

def plot_confusion_matrices(labels, preds, name_to_save: str, path_to_save: str):
    acc = accuracy_score(labels, preds)
    conf = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(conf)
    plt.clf()
    disp.plot()
    plt.title(f'acc: {acc} | {name_to_save}')
    make_dir(path_to_save)
    plt.savefig(f'{path_to_save}/{name_to_save}.png')

def plot_conf(conf: np.ndarray, name_to_save: str, path_to_save: str):
    make_dir(path_to_save)
    np.fill_diagonal(conf, 0)
    conf = (np.tril(conf) + np.triu(conf).T).astype(int)

    # as type int
    disp = ConfusionMatrixDisplay(conf)
    plt.clf()
    disp.plot()
    plt.title(f'{name_to_save}')
    make_dir(path_to_save)
    plt.savefig(f'{path_to_save}/{name_to_save}.png')
    
def cosine_similarity(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    dot_product = np.dot(v1, v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

def get_degree_cosine(weights, means, zero_diag: bool = False): 
    N = weights.shape[0]
    cosine_matrix = np.zeros((N, N))
    angle_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if zero_diag and i == j: 
                cosine_matrix[i, j] = 0.0
                angle_matrix[i, j]  = 0.0
            else: 
                cosine_matrix[i, j] = cosine_similarity(weights[i], means[j])
                angle_matrix[i, j]  = np.arccos(cosine_matrix[i, j]) * (180.0 / np.pi)
            
    return angle_matrix, cosine_matrix

def custom_plot_features(features, labels, weights, title: str, coeff: int, path: str='images/feats', save_only=True):  
    plt.clf()
    # plt.set_aspect('equal', adjustable='box')
    fig, ax = plt.subplots()
    fig.set_dpi(150)
    fig.set_size_inches(12, 10)

    # ax.set_aspect('equal', 'box')
    
    # ax.set_figure(figsize=(12, 10), dpi=150)

    c = ['black', '#0000ff', '#990000', '#00ffff', '#ffff00',
                        '#ff00ff', '#009900', '#999900', '#00ff00', '#009999']    
  
    if features.shape[0] < 2000: marker_size = 4
    elif features.shape[0] < 11000: marker_size = 3
    else: marker_size = 1

    for class_num in range(10):
        ax.plot(features[labels==class_num, 0], 
                    features[labels==class_num, 1],
                    '.', ms=marker_size, c=c[class_num], alpha=0.4, label=class_num)

        ax.plot([0, weights[class_num, 0]*coeff], [0, weights[class_num, 1]*coeff],
                    c='black', lw=4, label=class_num)

        ax.plot([0, weights[class_num, 0]*coeff], [0, weights[class_num, 1]*coeff],
                    c=c[class_num], lw=3, label=class_num)
        
        ax.annotate(f'{class_num}', [weights[class_num, 0]*coeff, weights[class_num, 1]*coeff])
        
    # ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right', markerscale=3)
    ax.legend()
    ax.set_title(title)
    
    # ax.ylim(-200, 200)
    # ax.xlim(-200, 200)
    ax.axis('square')
 
    make_dir(path)
    plt.savefig(f'{path}/{title}.jpg')
    
    if not save_only:
        plt.show()
    
    plt.clf()

def plot_first_last_weights(weights1, weights2, path: str, plot_title: str):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    c = ['black', '#0000ff', '#990000', '#00ffff', '#ffff00',
        '#ff00ff', '#009900', '#999900', '#00ff00', '#009999']    
    
    for class_num in range(10):
        coeff = 1

        ax1.plot([0, weights1[class_num, 0]*coeff], [0, weights1[class_num, 1]*coeff],
                    c=c[class_num], lw=3, label=class_num)
        ax1.annotate(f'{class_num}', [weights1[class_num, 0]*coeff, weights1[class_num, 1]*coeff],
                     fontsize=10)
        ax1.set_title(f'{plot_title} - start of epoch')
        ax1.axis('square')
        # ax1.legend(markerscale=3.0)

        ax2.plot([0, weights2[class_num, 0]*coeff], [0, weights2[class_num, 1]*coeff],
                    c=c[class_num], lw=3, label=class_num)
        ax2.annotate(f'{class_num}', [weights2[class_num, 0]*coeff, weights2[class_num, 1]*coeff],
                     fontsize=10)
        ax2.set_title(f'{plot_title} - end of epoch')
        ax2.axis('square')
        # ax2.legend(markerscale=3.0)

    make_dir(path)
    plt.savefig(f'{path}/{plot_title}.jpg')    
    plt.clf()

def custom_plot_weights(weights, plot_title: str, path: str='images', save_only=True):  
    plt.clf()
    # plt.figure(figsize=(12, 10), dpi=150)
    
    fig, ax = plt.subplots()
    # ax.set_aspect('equal', 'box')

    c = ['black', '#0000ff', '#990000', '#00ffff', '#ffff00',
                        '#ff00ff', '#009900', '#999900', '#00ff00', '#009999']    
  
    for class_num in range(10):
        coeff = 1
        ax.plot([0, weights[class_num, 0]*coeff], [0, weights[class_num, 1]*coeff],
                    c=c[class_num], lw=3, label=class_num)
        ax.annotate(f'{class_num}', [weights[class_num, 0]*coeff, weights[class_num, 1]*coeff],
                     fontsize=10)
        
    ax.legend(markerscale=3.0)
    ax.set_title(plot_title)
    # ax.set_xticks(fontsize=15)
    # ax.set_yticks(fontsize=15)
    ax.axis('square')

    make_dir(path)
    plt.savefig(f'{path}/{plot_title}.jpg')
    
    if not save_only:
        plt.show()
    
    plt.clf()


def custom_plot_loss(agg_dict, keys_to_plot: List[str], title: str, name_to_save: str,  runPath: str,  annotate_min: bool=True): 
    has_all_keys = all(elem in list(agg_dict.keys()) for elem in keys_to_plot)
    if not has_all_keys:
        print('invalid key for loss')
        # print(list(agg_dict.keys()))
        print(keys_to_plot)
        return
    
    '''Maximum Two plots, # TODO change to 4 plots
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=[12, 7], dpi=100)
    
    metric_names = keys_to_plot
    plot_indx = 0
    
    for i in range(min(len(ax), len(metric_names))):
        col = ax[i]
        metric_name = metric_names[plot_indx]
    
        ann_x = np.argmin(np.array(agg_dict[metric_name])) if annotate_min else np.argmax(np.array(agg_dict[metric_name]))
        ann_y = agg_dict[metric_name][ann_x]
        
        col.annotate("{:.8f}".format(ann_y), [ann_x, ann_y])
        col.plot(agg_dict[metric_name], '-x', label=f'{metric_name}', markevery = [ann_x])

        col.set_xlabel(xlabel='iterations/epochs')
        col.set_ylabel(ylabel=f'{metric_name}')

        col.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        col.legend()
        # col.label_outer()

        plot_indx += 1

    fig.suptitle(f'{title}')

    plt.savefig(f'{runPath}/{name_to_save}.jpg')
    plt.clf()
    

def visualize_10D(feat, labels, epoch, path, num_classes, name_to_save: str = None): 
    c = ['black', '#0000ff', '#990000', '#00ffff', '#ffff00',
         '#ff00ff', '#009900', '#999900', '#00ff00', '#009999']
    
    plt.clf()
    plt.figure(figsize=(20, 15), dpi=90)
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.1, hspace=0.1)
    plot_idx = 1
    for i in range(10):
        for j in range(i + 1, 10):
            plt.subplot(5, 9, plot_idx)
            plot_idx += 1

            for l in range(num_classes):
                #idxs = y_te == l
                #plt.plot(XU_te[idxs, i], XU_te[idxs, j], 'o', color=colors[l], alpha=0.5)
                plt.plot(feat[labels == l, i], feat[labels == l, j], '.', ms=1, c=c[l], alpha=0.5)
            plt.axis('off')
            plt.axis('equal')
            
    
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], markerscale=5.0)
    plt.text(0, 0, f"epoch={epoch}")
    make_dir(path)
    if name_to_save is None: 
        plt.savefig(f'{path}/epoch=%d.jpg' % epoch)
    else:
        plt.savefig(f'{path}/{name_to_save}-{epoch}.jpg')
    plt.clf()

def visualize(feat, labels, epoch, path, num_classes):
    # # plt.ion()
    # c = ['#ff0000', '#0000ff', '#990000', '#00ffff', '#ffff00',
    #      '#ff00ff', '#009900', '#999900', '#00ff00', '#009999']
    c = ['black', '#0000ff', '#990000', '#00ffff', '#ffff00',
         '#ff00ff', '#009900', '#999900', '#00ff00', '#009999']
    
    plt.clf()
    for i in range(num_classes):
        if feat.shape[0] < 10000: marker_size = 3
        else: marker_size = 1
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', ms=marker_size, c=c[i])
        # plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])

    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right', markerscale=5.0)
    plt.text(0, 0, "epoch=%d" % epoch)
    plt.savefig(f'{path}/epoch=%d.jpg' % epoch)
    plt.clf()


def save_gif(num_frames, path_of_frames, name, fps=10):
    frames = []
    for t in range(num_frames):
        #
        image = imageio.imread(f'{path_of_frames}/epoch=%d.jpg' % t)
        frames.append(image)

    imageio.mimsave(f'{path_of_frames}/{name}.gif',  # output gif
                    frames,  # array of input frames
                    fps=fps,  # optional: frames per second
                    # loop=1
                    )


def save_var(c_vars, name: str, num_classes: int): 
    pass

def plot_save_variances(results, path_to_save, phase='train'):
    loss_func_strings = list(results.keys())
    num_models = len(loss_func_strings)
    num_classes = results[loss_func_strings[0]]['num_classes']

    # not shared axes:
    fig, axes = plt.subplots(num_classes, num_models, figsize=(30, 20))
    # shared axes:
    # fig, axes = plt.subplots(num_classes, num_models, figsize=(30, 20), sharey='row')
    if len(axes.shape) == 1: axes = axes[:, None]

    subplot_indx = 1
    plot_exists = False
    for loss_indx, model_loss in enumerate(loss_func_strings):
        if phase in results[model_loss]['phase_list']:
            plot_exists = True
            c_vars = results[model_loss][phase]['c_var']  # [num_epochs, c, d]

            # c_vars --> [epoch][c, d]
            num_epochs = c_vars.shape[0]
            # num_classes= c_vars.shape[1]
            ### %
            # (c_vars.shape)

            dimens = c_vars.shape[2]

            for c in range(num_classes):
                for d in range(dimens):
                    # plot c, d
                    # plt.subplot(num_classes, num_models, subplot_indx)
                    # plt.plot(c_vars[:, c, d], label =f'var dim {d}')
                    axes[c, loss_indx].plot(c_vars[:, c, d], label=f'var dim {d}')

                # plt.title(f"c={c} {model_loss}", fontsize=20)
                # plt.xticks(fontsize=14)
                # plt.yticks(fontsize=14)
                axes[c, loss_indx].title.set_text(f"c={c} {re.split('=| ', model_loss)[3]} {re.split('=| ', model_loss)[5]}")
                # axes[c, loss_indx].set_xlabel(fontsize=14)
                # axes[c, loss_indx].set_ylabel(fontsize=14)

                # plt.legend(bbox_to_anchor =(0.75, 1.15), ncol = dimens)
                # axes[c, loss_indx].set_legend(bbox_to_anchor =(0.75, 1.15), ncol = dimens)
                subplot_indx += num_models

            subplot_indx = loss_indx + 2

    plt.legend(loc='best')
    # plt.tight_layout()
    plt.suptitle(f"Variances")
    if plot_exists:
        plt.savefig(f"{path_to_save}/{phase}-Variances.png")

    plt.clf()
    return


def plot_save_accs(results, path_to_save, phase='train'):
    # ^? 
    # allowed to call .subplots() everytime i want to plot sth new?  
    plt.subplots(figsize=(20, 20))

    # loss_func_strings = list(results.keys())
    loss_func_strings = [results['name']]
    num_models = len(loss_func_strings)

    if num_models == 1:
        rows_num = 1
        cols_num = 1
    elif num_models == 2:
        rows_num = 1
        cols_num = 2
    elif num_models <= 4:
        rows_num = 2
        cols_num = 2
    elif num_models <= 6:
        rows_num = 2
        cols_num = 3
    elif num_models <= 8:
        rows_num = 2
        cols_num = 4
    else:
        rows_num = 2
        cols_num = 7

    subplot_indx = 1
    plot_exists = False
    for loss_indx, model_loss in enumerate(loss_func_strings):
        if phase in results[model_loss]['phase_list']:
            plot_exists = True
            train_acc_history = results[model_loss][phase]['acc']  # [num_ecohcs, ]
            num_epochs = len(train_acc_history)

            plt.subplot(rows_num, cols_num, subplot_indx)
            plt.plot(train_acc_history, label=f'{model_loss}')
            plt.title(f"Acc {phase}-{model_loss}", fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            subplot_indx += 1

    # plt.legend()
    if plot_exists:
        plt.savefig(f'{path_to_save}/{phase}-train_acc.png')

    plt.clf()
    return

def plot_save_acc(phase_list: List[str], accs: Dict[str, Dict[str, np.ndarray]],
                   root_to_save: str, name_to_save: str) -> None:
    """
    accs -> {
     'experiment_name': dict{
          '{phase[0]}': [number_of_experiments][num_epochs][1],
          '{phase[1]}': [number_of_experiments][num_epochs][1],     
      }
    } 
    """
    plt.subplots(figsize=(16, 8))
    exps = list(accs.keys())
    num_exps = len(exps)
    rows_num, cols_num = get_rows_cols_num(num_exps)
    subplot_indx = 1
    
    for experiment_name in exps:  
        plt.subplot(rows_num, cols_num, subplot_indx)
        for phase in phase_list:     
            plt.plot(accs[experiment_name][phase], label=f'acc - {experiment_name}-{phase}')
            
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()
        subplot_indx +=1
        
    plt.savefig(f'{root_to_save}/{name_to_save}.png')
    plt.clf()

    return


def plot_save_cross_cov(results, path_to_save, phase='train'):
    loss_func_strings = list(results.keys())
    num_classes = results[loss_func_strings[0]]['num_classes']
    num_models = len(loss_func_strings)

    # not shared axes:
    # fig, axes = plt.subplots(num_classes, num_models, figsize=(5, 5))
    # shared axes:
    fig, axes = plt.subplots(num_classes, num_models, figsize=(10, 7), sharey='row')

    if (len(axes.shape) == 1): axes = axes[:, None]

    subplot_indx = 1
    plot_exists = False
    for loss_indx, model_loss in enumerate(loss_func_strings):
        if phase in results[model_loss]['phase_list']:
            plot_exists = True
            cross_cov = results[model_loss][phase]['cross_cov']  # [num_epochs, c]
            num_epochs = cross_cov.shape[0]
            num_classes = cross_cov.shape[1]

            for c in range(num_classes):
                # plt.subplot(num_classes, num_models, subplot_indx)
                # plt.plot(cross_cov[:, c])
                # plt.title(f"c={c} {model_loss}", fontsize=20)
                # plt.xticks(fontsize=15)
                # plt.yticks(fontsize=15)
                axes[c, loss_indx].plot(cross_cov[:, c])
                axes[c, loss_indx].title.set_text(f"c={c} {re.split('=| ', model_loss)[3]} {re.split('=| ', model_loss)[5]}")
               
                ## more tick freq + grid
                start, end = axes[c, loss_indx].get_ylim()
                axes[c, loss_indx].yaxis.set_ticks(np.arange(start, end, 0.9))
                axes[c, loss_indx].grid(True, axis='both')
                subplot_indx += num_models

            subplot_indx = loss_indx + 2

    # plt.legend(loc='best')
    plt.tight_layout()
    plt.suptitle(f"{phase}-Cross cov")
    
    if plot_exists:
        plt.savefig(f"{path_to_save}/{phase}-Cross_covs.png")
    
    plt.clf()
    return


def plot_save_loss(phase_list: List[str], losses: Dict[str, Dict[str, np.ndarray]],
                   root_to_save: str, name_to_save: str) -> None:
    """
    losses -> {
     'experiment_name': dict{
          '{phase[0]}': [number_of_experiments][num_epochs][1],
          '{phase[1]}': [number_of_experiments][num_epochs][1],     
      }
    } 
    """
    plt.subplots(figsize=(8, 4))
    exps = list(losses.keys())
    num_exps = len(exps)
    rows_num, cols_num = get_rows_cols_num(num_exps)
    subplot_indx = 1
    
    for experiment_name in exps:  
        plt.subplot(rows_num, cols_num, subplot_indx)
        # plt.title(f"Loss - {phase}- {model_loss}", fontsize=20)
        for phase in phase_list:     
            plt.plot(losses[experiment_name][phase], label=f'{experiment_name}-{phase}')
            
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()
        subplot_indx +=1
        
    plt.savefig(f'{root_to_save}/{name_to_save}.png')
    plt.clf()

    return

# change this functions so they just get a list/np array as input, 
# they don't have to get the whole dictionary
def plot_save_losses(results, path_to_save, phase='train'):
    # ^? 
    # allowed to call .subplots() everytime i want to plot sth new?  
    plt.subplots(figsize=(20, 20))

    # loss_func_strings = list(results.keys())
    loss_func_strings = [results['name']]
    num_models = len(loss_func_strings)

    if num_models == 1:
        rows_num = 1
        cols_num = 1
    elif num_models == 2:
        rows_num = 1
        cols_num = 2
    elif num_models <= 4:
        rows_num = 2
        cols_num = 2
    elif num_models <= 6:
        rows_num = 2
        cols_num = 3
    elif num_models <= 8:
        rows_num = 2
        cols_num = 4
    else:
        rows_num = 2
        cols_num = 7

    subplot_indx = 1
    plot_exists = False
    for loss_indx, model_loss in enumerate(loss_func_strings):
        
        if phase in results[model_loss]['phase_list']:
            plot_exists = True
            loss_history = results[model_loss][phase]['loss']  # [num_ecohcs, 1]
            num_epochs = len(loss_history)

            plt.subplot(rows_num, cols_num, subplot_indx)
            plt.plot(loss_history, label=f'{model_loss}')
            plt.title(f"Loss - {phase}- {model_loss}", fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            subplot_indx += 1

    # plt.legend()
    if plot_exists:
        plt.savefig(f'{path_to_save}/{phase}-Losses.png')\
        
    plt.clf()
    return


def func_class_list_to_string_list(func_list):
    results = []
    for f in func_list:
        results.append(f.name)

    return results


def func_list_to_string_list(func_list):
    results = []
    for f in func_list:
        if inspect.isfunction(f):
            results.append(f.__name__)
        else:
            # if using partial to pass functions, call .func
            results.append(f.func.__name__)

    return results


def SetMarkerColor(x):
    # color_palette = ['#636EFA', '#FB0D0D', '#a80f8f', '#00FE35', '#266934']
    color_palette = ['#006400', '#00008b', '#ff4500', '#ff00ff',
                     '#ffff00', '#deb887', '#00ff00', '#00ffff',
                     '#6495ed', '#b03060']
    # color_palette = ['black', 'blue', 'red']

    return color_palette[x.astype(int)]


def SetMarkerSymbol(x):
    symbol_palette = ['triangle-down', 'cross', 'circle']
    return symbol_palette[x.astype(int)]


def ell_arc_2d(mean_list, cov, N=100):
    cx = mean_list[0]
    cy = mean_list[1]
    a = cov[0][0]
    b = cov[0][1]
    c = cov[1][0]
    d = cov[1][1]

    eigvals, eigvecs = np.linalg.eig(cov)
    eigval_indxs = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, eigval_indxs].T
    eigvals = eigvals[eigval_indxs]

    l1 = eigvals[0]
    l2 = eigvals[1]

    if b == 0 and a >= c:
        theta = 0
    elif b == 0 and a < c:
        theta = pi / 2
    else:
        theta = np.arctan2(l1 - a, l1 - b)

    t = np.linspace(0, 2 * pi, N)

    xt = np.sqrt(5.991 * l1) * cos(theta) * cos(t) - np.sqrt(5.991 * l2) * sin(theta) * cos(t) + cx
    yt = np.sqrt(5.991 * l1) * sin(theta) * cos(t) + np.sqrt(5.991 * l2) * cos(theta) * sin(t) + cy

    return xt, yt

def get_rows_cols_num(num_exps: int):
    if num_exps == 1:
        rows_num = 1
        cols_num = 1
    elif num_exps == 2:
        rows_num = 1
        cols_num = 2
    elif num_exps <= 4:
        rows_num = 2
        cols_num = 2
    elif num_exps <= 6:
        rows_num = 2
        cols_num = 3
    elif num_exps <= 8:
        rows_num = 2
        cols_num = 4
    else:
        rows_num = 2
        cols_num = 7

    return rows_num, cols_num

def adv_analysis(
        orig_features: np.ndarray, 
        adv_features: np.ndarray, 
        orig_preds: np.ndarray, 
        adv_preds: np.ndarray, 
        orig_labels: np.ndarray,
        adv_labels: np.ndarray,
        weights: np.ndarray,
        name_to_save: str, 
        path_to_save: str,
    ):
    feat_dim = orig_features.shape[1]
    # Create a 2x2 subplot grid with custom widths for columns
    fig = plt.figure(figsize=[20, 10], dpi=200)
    gs = gridspec.GridSpec(2, 4, width_ratios=[0.25, 0.25, 0.25, 0.25])
    """
    [0, 0] | [0, 1] | [0, 2] | [0, 3]
    [1, 0] | [1, 1] | [1, 2] | [1, 3]
    """

    """
    - w_i with mu_orig_j X
    - w_i with mu_adv_j X 
    - w_i with w_j X 
    - mu_orig_i with mu_orig_j X 
    - mu_adv_i with mu_adv_j X 
    - mu_orig_i with mu_adv_j X 
    - conf orig X 
    - conf adv X 
    """

    # Create a subplot that spans two rows (belongs to the first column)
    ax_w_mu_orig = plt.subplot(gs[0, 0])
    ax_w_mu_adv = plt.subplot(gs[1, 0])
    ax_w_w = plt.subplot(gs[0, 1])
    ax_mu_orig_mu_adv = plt.subplot(gs[1, 1])
    ax_mu_orig_mu_orig = plt.subplot(gs[0, 2])
    ax_mu_adv_mu_adv = plt.subplot(gs[1, 2])
    ax_conf_orig = plt.subplot(gs[0, 3])
    ax_conf_adv = plt.subplot(gs[1, 3])

    orig_mean_classes = []
    for class_num in range(len(np.unique(orig_labels))):
        feats_c = orig_features[orig_labels==class_num] # [k, 2]
        mean_c = np.mean(feats_c, axis=0) # [2, ]
        orig_mean_classes.append(mean_c)
    orig_mean_classes = np.vstack(orig_mean_classes)

    adv_mean_classes = []
    for class_num in range(len(np.unique(adv_labels))):
        feats_c = adv_features[adv_labels==class_num] # [k, 2]
        mean_c = np.mean(feats_c, axis=0) # [2, ]
        adv_mean_classes.append(mean_c)
    adv_mean_classes = np.vstack(adv_mean_classes)

    ######START###### conf orig 
    curr_ax = ax_conf_orig
    acc = accuracy_score(orig_labels, orig_preds)
    conf = confusion_matrix(orig_labels, orig_preds)

    disp = ConfusionMatrixDisplay(conf)
    disp.plot(ax=curr_ax, colorbar=False)
    curr_ax.set_title(f'Clean Acc: {acc}')
    ######END######

    ######START###### conf adv 
    curr_ax = ax_conf_adv
    acc = accuracy_score(adv_labels, adv_preds)
    conf = confusion_matrix(adv_labels, adv_preds)

    disp = ConfusionMatrixDisplay(conf)
    disp.plot(ax=curr_ax, colorbar=False)
    curr_ax.set_title(f'Adversarial Acc: {acc}')
    ######END######

    ######START###### w_i with mu_orig_j
    curr_ax = ax_w_mu_orig
    w_mu_angle_mat, _ = get_degree_cosine(weights, orig_mean_classes)
    curr_ax.set_title('w_mu_orig angles')
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    im = curr_ax.imshow(np.zeros((10, 10)))
    curr_ax.set_xticks(np.arange(len(classes)))
    curr_ax.set_yticks(np.arange(len(classes)))
    plt.setp(curr_ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = curr_ax.text(j, i, f'{w_mu_angle_mat[i, j]: 0.2f}',
                        ha="center", va="center", color="w", fontdict={'size': 8})
    ######END######

    ######START###### w_i with mu_adv_j
    curr_ax = ax_w_mu_adv
    w_mu_angle_mat, _ = get_degree_cosine(weights, adv_mean_classes)
    curr_ax.set_title('w_mu_adv angles')
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    im = curr_ax.imshow(np.zeros((10, 10)))
    curr_ax.set_xticks(np.arange(len(classes)))
    curr_ax.set_yticks(np.arange(len(classes)))
    plt.setp(curr_ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = curr_ax.text(j, i, f'{w_mu_angle_mat[i, j]: 0.2f}',
                        ha="center", va="center", color="w", fontdict={'size': 8})
    ######END######


    ######START###### W W 
    curr_ax = ax_w_w
    w_w_angle_mat, _ = get_degree_cosine(weights, weights, zero_diag=True)
    curr_ax.set_title('w_w angles')
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    im = curr_ax.imshow(np.zeros((10, 10)))
    curr_ax.set_xticks(np.arange(len(classes)))
    curr_ax.set_yticks(np.arange(len(classes)))
    plt.setp(curr_ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = curr_ax.text(j, i, f'{w_w_angle_mat[i, j]: 0.2f}',
                        ha="center", va="center", color="w", fontdict={'size': 8})
    ######END######

    ######START###### mu_orig_i with mu_adv_j
    curr_ax = ax_mu_orig_mu_adv
    mu_mu_angle_mat, _ = get_degree_cosine(orig_mean_classes, adv_mean_classes, zero_diag=False)
    curr_ax.set_title('MU(orig_adv) angles')
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    im = curr_ax.imshow(np.zeros((10, 10)))
    curr_ax.set_xticks(np.arange(len(classes)))
    curr_ax.set_yticks(np.arange(len(classes)))
    plt.setp(curr_ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    # print(angle_mat)
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = curr_ax.text(j, i, f'{mu_mu_angle_mat[i, j]: 0.2f}',
                        ha="center", va="center", color="w", fontdict={'size': 8})
    ######END######

    
    ######START###### mu_adv_i with mu_adv_j
    curr_ax = ax_mu_adv_mu_adv
    mu_mu_angle_mat, _ = get_degree_cosine(adv_mean_classes, adv_mean_classes, zero_diag=True)
    curr_ax.set_title('MU(adv_adv) angles')
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    im = curr_ax.imshow(np.zeros((10, 10)))
    curr_ax.set_xticks(np.arange(len(classes)))
    curr_ax.set_yticks(np.arange(len(classes)))
    plt.setp(curr_ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    # print(angle_mat)
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = curr_ax.text(j, i, f'{mu_mu_angle_mat[i, j]: 0.2f}',
                        ha="center", va="center", color="w", fontdict={'size': 8})
    ######END######

    ######START###### mu_orig_i with mu_orig_j
    curr_ax = ax_mu_orig_mu_orig
    mu_mu_angle_mat, _ = get_degree_cosine(orig_mean_classes, orig_mean_classes, zero_diag=True)
    curr_ax.set_title('MU(orig_orig) angles')
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    im = curr_ax.imshow(np.zeros((10, 10)))
    curr_ax.set_xticks(np.arange(len(classes)))
    curr_ax.set_yticks(np.arange(len(classes)))
    plt.setp(curr_ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    # print(angle_mat)
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = curr_ax.text(j, i, f'{mu_mu_angle_mat[i, j]: 0.2f}',
                        ha="center", va="center", color="w", fontdict={'size': 8})
    ######END######
        
    plt.tight_layout()
    make_dir(path_to_save)
    plt.suptitle(f'{feat_dim} adv agnles - {name_to_save}')
    plt.savefig(f'{path_to_save}/{name_to_save} angles.jpg')
    plt.clf()

    ################################################## HISTS W
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for i in range(len(axes)):
        if i >= 10:
            fig.delaxes(axes[i])

    for class_num in range(len(np.unique(orig_labels))):
        orig_feats_c = orig_features[orig_labels==class_num] 
        adv_feats_c  = adv_features[adv_labels==class_num]

        orig_angles = angles_between_samples_and_vector(orig_feats_c, weights[class_num])
        adv_angles = angles_between_samples_and_vector(adv_feats_c, weights[class_num])
        
        axes[class_num].hist(orig_angles, bins=180, range=(0.0, 180.0), label='Clean data', alpha=0.5) # density=True, 
        axes[class_num].hist(adv_angles, bins=180, range=(0.0, 180.0), label='Adverarial data', alpha=0.5) # density=True, 
        
        axes[class_num].set_title(f'{class_num}') 
        axes[class_num].set_xlabel('angle')
        axes[class_num].set_ylabel('freq')    
        axes[class_num].grid(True)
    
    plt.tight_layout()
    plt.legend()
    make_dir(path_to_save)
    plt.suptitle('angle between (adversarial and clean feature) and class weight')
    plt.savefig(f'{path_to_save}/{name_to_save} w_angles histogram.jpg')
    plt.clf()

    ################################################### HISTS MU
    # fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    # axes = axes.flatten()
    # for i in range(len(axes)):
    #     if i >= 10:
    #         fig.delaxes(axes[i])

    # for class_num in range(len(np.unique(labels))):
    #     orig_feats_c = orig_features[labels==class_num] 
    #     adv_feats_c  = adv_features[labels==class_num]

    #     orig_angles = angles_between_samples_and_vector(orig_feats_c, weights[class_num])
    #     adv_angles = angles_between_samples_and_vector(adv_feats_c, weights[class_num])
        
    #     axes[class_num].hist(orig_angles, bins=180, density=True, range=(0.0, 180.0), label='Clean data', alpha=0.5)
    #     axes[class_num].hist(adv_angles, bins=180, density=True, range=(0.0, 180.0), label='Adverarial data', alpha=0.5)
        
    #     axes[class_num].set_title(f'{class_num}') 
    #     axes[class_num].xlabel('angle')
    #     axes[class_num].ylabel('freq')    

    # plt.tight_layout()
    # plt.legend()
    # make_dir(path_to_save)
    # plt.suptitle('angle between (adversarial and clean feature) and class weight')
    # plt.grid(True)
    # plt.savefig(f'{path_to_save}/w_angles histogram-{name_to_save}.jpg')
    # plt.clf()

    ################################################## HISTS NORMS
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for i in range(len(axes)):
        if i >= 10:
            fig.delaxes(axes[i])

    diff_norms = []
    max_diff = None
    min_diff = None
    for class_num in range(len(np.unique(orig_labels))):
        orig_feats_c = orig_features[orig_labels==class_num] 
        adv_feats_c  = adv_features[adv_labels==class_num]

        orig_norms = np.linalg.norm(orig_feats_c, axis=1)
        adv_norms = np.linalg.norm(adv_feats_c, axis=1)
        
        diff_norms.append(adv_norms - orig_norms)
        max_diff = max(max_diff, np.max(diff_norms[-1])) if not max_diff is None else np.max(diff_norms[-1])
        min_diff = min(min_diff, np.min(diff_norms[-1])) if not min_diff is None else np.min(diff_norms[-1])

    for class_num in range(len(np.unique(orig_labels))):
        axes[class_num].hist(diff_norms[class_num], bins = 100, range=(min_diff, max_diff), label='adv_norms - orig_norms', alpha=0.75) # density=True, 
        axes[class_num].set_title(f'{class_num}') 
        axes[class_num].set_xlabel('diff norms')
        axes[class_num].set_ylabel('freq')    
        axes[class_num].grid(True)
    
    plt.tight_layout()
    plt.legend()
    make_dir(path_to_save)
    plt.suptitle('diff norm between adversarial and clean feature')
    plt.savefig(f'{path_to_save}/{name_to_save} diff norm hist.jpg')
    plt.clf()


def _norm_analysis(
        features, 
        labels, 
        weights, 
        centers = None):
    
    norm_w = np.linalg.norm(weights, axis=1) # [C, ]
    norm_centers = np.linalg.norm(centers, axis=1) # [C, ]

    mean_classes = []
    std_norms_feats = []
    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        feats_c = features[labels == i] # [m, d]
        norms_c = np.linalg.norm(feats_c, axis=1) # [m, ]
        std_norms_feats.append(np.std(norms_c)) # appends [1, ]
        mean_classes.append(np.mean(feats_c, axis=0)[None, :])
    mean_classes = np.concatenate(mean_classes)
    norm_mu = np.linalg.norm(mean_classes, axis=1) # [C, ]

    return { 
        'mean_norm_Ws': np.mean(norm_w),
        'std_norm_Ws': np.std(norm_w),
        'mean_norm_MUs': np.mean(norm_mu),
        'std_norm_MUS': np.std(norm_mu),
        'mean_within_std_norm_feats': np.mean(np.array(std_norms_feats)),
        'mean_norm_centers': None if centers is None else np.mean(norm_centers),
        'std_norm_centers': None if centers is None else np.std(norm_centers),
    }

def norm_analysis(
        features, 
        labels, 
        weights, 
        name_to_save: str, 
        path_to_save: str, 
        centers = None):
    make_dir(path_to_save)
    out = _norm_analysis(features=features, labels=labels, weights=weights, centers=centers)
    norm_w = np.linalg.norm(weights, axis=1)
    norm_centers = np.linalg.norm(centers, axis=1)

    mean_classes = []
    std_norms_classes = []
    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        feats_c = features[labels == i]
        norms_c = np.linalg.norm(feats_c, axis=1)
        std_norm_c = np.std(norms_c)
        std_norms_classes.append(std_norm_c)
        mean_classes.append(np.mean(feats_c, axis=0)[None, :])
    mean_classes = np.concatenate(mean_classes)

    norm_mu = np.linalg.norm(mean_classes, axis=1)

    fig = plt.figure(figsize=[10, 10]) # , dpi=200
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.5, 0.5])
    """
    [0, 0] | [0, 1] 
    [1, 0] | [1, 1] 
    """
    ax_norm_w = plt.subplot(gs[0, 0])
    ax_norm_mu = plt.subplot(gs[0, 1])
    ax_norm_centers = plt.subplot(gs[1, 0])

    ax_norm_w.bar(np.arange(num_classes), norm_w)
    ax_norm_w.set_title('Norm of weights')
    
    ax_norm_mu.bar(np.arange(num_classes), norm_mu, yerr=std_norms_classes)
    ax_norm_mu.set_title('Norm of mean of features')

    if not centers is None:
        ax_norm_centers.bar(np.arange(num_classes), norm_centers)
        ax_norm_centers.set_title('Norm of mean of centers')
    
    plt.savefig(f'{path_to_save}/{name_to_save}.png')
    plt.clf()

    return { 
        'mean_norm_w': np.mean(norm_w),
        'std_norm_w': np.std(norm_w),
        'mean_norm_mu': np.mean(norm_mu),
        'std_norm_mu': np.std(norm_mu),
        'mean_std_feats': np.mean(np.array(std_norms_classes)),
        'mean_norm_centers': None if centers is None else np.mean(norm_centers),
        'std_norm_centers': None if centers is None else np.std(norm_centers),
    }
