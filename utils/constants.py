import math


class Constants(object):
    mnist_normalization = {
        "mean": [
            0.1307,
        ],
        "std": [
            0.3081,
        ],
    }
    cifar10_normalization = {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    }

    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0

    # Alireza 
    OUTPUT_DIR = 'D:/Uni/NN/Project/Neural_Network_Project/experiments'
    # MNIST_DIR = '/home/hamed/EBV/LDA-FUM/LDA-FUM/data/MNIST'
    # CIFAR10_DIR = '/home/hamed/EBV/LDA-FUM/LDA-FUM/data/CIFAR10'


