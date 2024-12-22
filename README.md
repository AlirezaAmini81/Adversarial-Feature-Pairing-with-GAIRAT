# AFP_GAIRAT: Adversarial Feature Pairing with Geometry-Aware Instance-Reweighted Adversarial Training

This repository contains the implementation and evaluation of AFP_GAIRAT, a novel method that integrates Adversarial Feature Pairing (AFP) with Geometry-Aware Instance-Reweighted Adversarial Training (GAIRAT) (https://openreview.net/forum?id=iAX0l6Cz8ub) (ICLR oral)<br/> to enhance the adversarial robustness of neural networks.

## Overview

AFP_GAIRAT combines the strengths of GAIRAT and a unique approach called Adversarial Feature Pairing (AFP) to create a more robust defense against adversarial attacks. While GAIRAT focuses on assigning instance-specific weights based on their geometric properties, AFP ensures that the features of natural and adversarial examples remain close. Together, these methods form AFP_GAIRAT, which achieves superior adversarial robustness.

## Contents

- Report.pdf: Detailed explanation of the AFP_GAIRAT method, experimental setup, and results.
- Code: Python implementation of the AFP_GAIRAT method.
- Experiments: Scripts and configurations used for running the experiments on the MNIST and CIFAR-10 datasets.
- Results: Output and analysis of the experiments, including accuracy and adversarial robustness metrics.

## Method: AFP_GAIRAT

### Aderserial Training

Adversarial training employs adversarial data, instead of clean data, for updating the models. 


### Adversarial Feature Pairing (AFP)

AFP aims to minimize the feature space distance between natural and adversarial examples. This method adds a term to the loss function to ensure that the features of natural and adversarial examples are paired closely together, enhancing the model's robustness to adversarial attacks.

### Geometry-Aware Instance-Reweighted Adversarial Training (GAIRAT)

GAIRAT assigns weights to training instances based on their distance from the decision boundary. The distances are approximated by the number of PGD steps that the PGD method requires to generate its misclassified adversarial variant.
Instances closer to the boundary receive higher weights, making the model focus more on these critical examples during training.

<p align="center">
    <img src="images/GAIRAT_learning_obj.png" width="800"\>
</p>
<p align="left">
The illustration of GAIRAT. GAIRAT explicitly gives larger weights on the losses of adversarial data (larger red), whose natural counterparts are closer to the decision boundary (lighter blue). GAIRAT explicitly gives smaller weights on the losses of adversarial data (smaller red), whose natural counterparts are farther away from the decision boundary (darker blue). </p>

### Key Components

- Instance Reweighting: Instances are weighted according to their distance from the decision boundary, as in GAIRAT.
- Feature Pairing: The features for natural and adversarial examples are paired to minimize their difference, enhancing robustness.

## Results

### MNIST

- AFP_GAIRAT: Demonstrates significant improvements in adversarial robustness while maintaining competitive natural accuracy.

### CIFAR-10

- AFP_GAIRAT: Shows substantial enhancement in robustness against adversarial attacks, outperforming other methods in robustness metrics.


### Preferred Prerequisites

* Python (3.6)
* Pytorch (1.2.0)
* CUDA
* numpy
* foolbox

### How to Use

```bash

```



