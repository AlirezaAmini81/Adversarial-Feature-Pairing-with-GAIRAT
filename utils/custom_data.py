import os
import sys
import torch
import random
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader, Sampler

# relative import hacks (sorry)
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # for bash user
os.chdir(parentdir)  # for pycharm user
import matplotlib.gridspec as gridspec

from utils import constants


def get_custom_data(data_loader_seed):
    input_size, num_classes, ds_train, ds_test, num_samples = load_data()

    batch_size = num_samples  # FULL BATCH LEARNING
    # batch_size = 128

    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(data_loader_seed)
        random.seed(data_loader_seed)

    g = torch.Generator()
    g.manual_seed(data_loader_seed)
    random.seed(data_loader_seed)
    torch.manual_seed(data_loader_seed)
    np.random.seed(data_loader_seed)

    full_data_loaders = {
        "train": DataLoader(
            ds_train,
            batch_size=batch_size,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True,
            shuffle=True,
            num_workers=0,
        ),
        "test": DataLoader(
            ds_test,
            batch_size=batch_size,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True,
            shuffle=False,
            num_workers=0,
        ),
    }

    return full_data_loaders, input_size, num_classes, batch_size


from utils.gcn_whitened_cifar10 import gcn_zca_CIFAR10
from utils.np_transforms import RandomHorizontalFlip


class UniformRandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    replacement: bool

    def __init__(
        self, data_source, batch_size: int, num_samples=None, generator=None
    ) -> None:
        self.data_source = data_source
        self.replacement = False  #
        self._num_samples = num_samples
        self.generator = generator
        self.batch_size = batch_size

        if not isinstance(self.replacement, bool):
            raise TypeError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=generator
                ).tolist()
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=generator,
            ).tolist()

        else:
            labels = torch.tensor(self.data_source.targets)

            # get indicies of each class
            num_classes = len(torch.unique(labels))
            class_indices = []

            for i in range(num_classes):
                curr_class_indicies = torch.where(labels == i)[0]
                rand_indices = torch.randperm(
                    curr_class_indicies.shape[0], generator=generator
                )
                class_indices.append(curr_class_indicies[rand_indices])

                # print('len', len(class_indices[-1]))

            least_freq = self.batch_size // num_classes
            num_least_freq_batches = self.num_samples // (least_freq * num_classes)
            # print(least_freq, num_least_freq_batches)

            final_indices = []
            for b in range(num_least_freq_batches + 1):
                # print('picking this indices from each class: ', b*least_freq, min(len(class_indices[-1]), (b+1)*least_freq) )
                batch_indicies = []
                for c in range(num_classes):
                    batch_indicies.append(
                        class_indices[c][
                            b
                            * least_freq : min(
                                len(class_indices[c]), (b + 1) * least_freq
                            )
                        ]
                    )

                batch_indicies = torch.cat(batch_indicies)
                final_indices.append(
                    batch_indicies[
                        torch.randperm(batch_indicies.shape[0], generator=generator)
                    ]
                )

            final_indices = torch.cat(final_indices)

            # print(labels.shape)
            # print(final_indices.shape, final_indices[0:10])

            # temp1 = torch.arange(labels.shape[0])
            # temp2 = final_indices.sort()[0]
            # print('all indices used? ', torch.sum(torch.abs(temp1-temp2)))

            yield from final_indices.tolist()
            # yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples


def get_gcn_zca_cifar10_data(
    data_loader_seed: int,
    uniform_sampler: bool,
    batch_size=128,
):
    # if augment:
    train_transform = torchvision.transforms.Compose(
        [
            # torchvision.transforms.ToPILImage(),
            # torchvision.transforms.RandomHorizontalFlip(p=0.5),
            # torchvision.transforms.ToTensor(),
            # torchvision.transforms.RandomHorizontalFlip(p=0.5),
            # torchvision.transforms.ConvertImageDtype(dtype=torch.float32),
            RandomHorizontalFlip(prob=0.5),
        ]
    )

    test_transform = torchvision.transforms.Compose(
        [
            # torchvision.transforms.ToPILImage(),
            # torchvision.transforms.ToTensor(),
            # torchvision.transforms.ConvertImageDtype(dtype=torch.float32),
        ]
    )

    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(data_loader_seed)
        random.seed(data_loader_seed)

    g = torch.Generator()
    g.manual_seed(data_loader_seed)
    random.seed(data_loader_seed)
    torch.manual_seed(data_loader_seed)
    np.random.seed(data_loader_seed)

    # saleh_dir = ''
    hamed_dir = "/home/hamed/Documents/pylearn_datasets/cifar10/pylearn2_gcn_whitened"
    robotics_dir = "/home/ramin/LDA-FUM/data/CIFAR10_GCN_ZCA"

    # if v2:
    #     hamed_dir += '_v2'
    #     robotics_dir += '_v2'

    # colab_dir =  ''
    trainset = gcn_zca_CIFAR10(root=robotics_dir, train=True, transform=train_transform)
    print("train set", trainset)

    testset = gcn_zca_CIFAR10(root=robotics_dir, train=False, transform=test_transform)
    print("test set", testset)

    if uniform_sampler:
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            sampler=UniformRandomSampler(trainset, batch_size),
            worker_init_fn=seed_worker,
            generator=g,
        )
        testloader = DataLoader(
            testset,
            batch_size=500,  #
            sampler=UniformRandomSampler(testset, 500),  #
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True,
            shuffle=True,
        )
        testloader = DataLoader(
            testset,
            batch_size=batch_size,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True,
            shuffle=False,
        )

    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    num_classes = 10
    input_size = 3 * 32 * 32

    full_data_loaders = {
        "train": trainloader,
        "test": testloader,
    }
    return full_data_loaders, input_size, num_classes, batch_size


def get_cifar10_data(
    data_loader_seed: int,
    uniform_sampler: bool,
    batch_size=128,
    adv_transform=False,
    resnet_transform=False,
    drop_last=False,
    double_test_batch_size=False,
    augment: str = "default",
    shuffle=True,
):

    if adv_transform:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    elif resnet_transform:
        train_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=240
                ),  # Resize to a larger size while maintaining aspect ratio
                transforms.RandomHorizontalFlip(),  # 50% chance of horizontal flip
                transforms.RandomCrop(size=224, padding=8),  # Random crop to 224x224
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=240
                ),  # Resize to a larger size while maintaining aspect ratio
                transforms.RandomHorizontalFlip(),  # 50% chance of horizontal flip
                transforms.RandomCrop(size=224, padding=8),  # Random crop to 224x224
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
    else:
        if augment == "default":
            train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        constants.Constants.cifar10_normalization["mean"],
                        constants.Constants.cifar10_normalization["std"],
                    ),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        constants.Constants.cifar10_normalization["mean"],
                        constants.Constants.cifar10_normalization["std"],
                    ),
                ]
            )
        elif augment == "madry":
            train_transform = transforms.Compose(
                [
                    transforms.Pad(4, padding_mode="reflect"),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        constants.Constants.cifar10_normalization["mean"],
                        constants.Constants.cifar10_normalization["std"],
                    ),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        constants.Constants.cifar10_normalization["mean"],
                        constants.Constants.cifar10_normalization["std"],
                    ),
                ]
            )

    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(data_loader_seed)
        random.seed(data_loader_seed)

    g = torch.Generator()
    g.manual_seed(data_loader_seed)
    random.seed(data_loader_seed)
    torch.manual_seed(data_loader_seed)
    np.random.seed(data_loader_seed)

    trainset = torchvision.datasets.CIFAR10(
        root=f"{constants.Constants.CIFAR10_DIR}",
        train=True,
        download=True,
        transform=train_transform,
    )
    print("train set", trainset)

    testset = torchvision.datasets.CIFAR10(
        root=f"{constants.Constants.CIFAR10_DIR}",
        train=False,
        download=True,
        transform=test_transform,
    )
    print("test set", testset)

    if uniform_sampler:
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            sampler=UniformRandomSampler(trainset, batch_size),
            worker_init_fn=seed_worker,
            generator=g,
        )
        testloader = DataLoader(
            testset,
            batch_size=batch_size if not double_test_batch_size else batch_size * 2,
            sampler=UniformRandomSampler(testset, batch_size),
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=drop_last,
            shuffle=shuffle,
        )
        testloader = DataLoader(
            testset,
            batch_size=batch_size if not double_test_batch_size else batch_size * 2,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=drop_last,
            shuffle=False,
        )
        

    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    num_classes = 10
    input_size = 3 * 32 * 32

    full_data_loaders = {
        'train': trainloader,
        'test':  testloader,
        'val': testloader,
    }
    return full_data_loaders, input_size, num_classes, batch_size


def get_selected_mnist_data(
    data_loader_seed: int,
    batch_size=128,
    selected_classes=list(np.arange(10)),
    tr_10k: bool = False,
):
    trainset = torchvision.datasets.MNIST(
        "MNIST",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    testset = torchvision.datasets.MNIST(
        "MNIST",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    train_idxs = torch.where(
        torch.isin(trainset.targets, torch.asarray(selected_classes))
    )[0]
    test_idxs = torch.where(
        torch.isin(testset.targets, torch.asarray(selected_classes))
    )[0]

    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(data_loader_seed)
        random.seed(data_loader_seed)

    g = torch.Generator()
    g.manual_seed(data_loader_seed)

    tr_size = train_idxs.shape[0]
    te_size = test_idxs.shape[0]

    if tr_10k:
        indices = torch.arange(1024 * 10)
        tr_size = 1024 * 10
        trainset = torch.utils.data.Subset(trainset, indices)

    # have to set shuffle to False, since SubsetRandomSampler already shuffles data
    full_data_loaders = {
        "train": DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=False,  #
            worker_init_fn=seed_worker,
            generator=g,
            sampler=SubsetRandomSampler(train_idxs, g),
            drop_last=True,
        ),
        "train_size": tr_size,
        "test": DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            sampler=SubsetRandomSampler(test_idxs),
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        "test_size": te_size,
    }

    input_size = 28 * 28
    num_classes = 10

    return full_data_loaders, input_size, num_classes, batch_size


def get_mnist_data(
    data_loader_seed,
    uniform_sampler: bool,
    batch_size=60000,
    drop_last=True,
    normalize=True,
    validation=False,
    shuffle=True,
):
    validationset = None
    if normalize:
        trnsfrms = (
            torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (constants.Constants.mnist_normalization["mean"][0],),
                        (constants.Constants.mnist_normalization["std"][0],),
                    ),
                ]
            )
            if normalize == True
            else torchvision.transforms.Compose([])
        )
    else:
        trnsfrms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

    trainset = torchvision.datasets.MNIST(
        f"{constants.Constants.MNIST_DIR}",
        download=True,
        train=True,
        transform=trnsfrms,
    )
    if validation:
        g_val = torch.Generator().manual_seed(42)
        trainset, validationset = torch.utils.data.random_split(
            trainset, [55000, 5000], generator=g_val
        )

    testset = torchvision.datasets.MNIST(
        f"{constants.Constants.MNIST_DIR}",
        train=False,
        download=True,
        transform=trnsfrms,
    )

    print("trainset: ", trainset)
    print("validationset: ", testset)
    print("testset: ", testset)

    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(data_loader_seed)
        random.seed(data_loader_seed)

    g = torch.Generator()
    g.manual_seed(data_loader_seed)
    random.seed(data_loader_seed)
    torch.manual_seed(data_loader_seed)
    np.random.seed(data_loader_seed)

    full_data_loaders = {
        "train": DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=drop_last,
        ),
        "test": DataLoader(
            testset,
            batch_size=batch_size,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=drop_last,
            shuffle=False,
        ),
        "val": (
            None
            if validationset is None
            else DataLoader(
                validationset,
                batch_size=batch_size,
                worker_init_fn=seed_worker,
                generator=g,
                drop_last=drop_last,
                shuffle=False,
            )
        ),
    }

    print(batch_size)
    input_size = 28 * 28
    num_classes = 10

    return full_data_loaders, input_size, num_classes, batch_size


class multi_var(Dataset):
    def __init__(
        self, mean_lists, cov_mats, size_list, num_classes, np_random_seed=None
    ):
        """
        Args:
            list of mean_list for each class id (sequence):       desired mean of data along each axis
            list of cov_mat for each class id (2d squared seq):   covariance matrix
            size (sequence):                                      size of each class
        """

        self.mean_lists = mean_lists
        self.cov_mats = cov_mats
        self.size_list = size_list
        self.num_classes = num_classes
        self.dim = len(mean_lists[0])
        assert num_classes == len(mean_lists), "num_classes != len(mean_lists)"
        assert num_classes == len(cov_mats), "num_classes != len(cov_mats)"
        assert num_classes == len(size_list), "num_classes != len(size)"
        self.data_matrix = None

        if np_random_seed != None:
            np.random.seed(np_random_seed)

        for c in range(num_classes):
            curr_class_labels = (np.ones((self.size_list[c])) * c).astype(int)
            curr_class_data_mat = np.vstack(
                (
                    np.random.multivariate_normal(
                        self.mean_lists[c], self.cov_mats[c], self.size_list[c]
                    ).T,
                    curr_class_labels,
                )
            ).T

            if c == 0:
                self.data_matrix = curr_class_data_mat
            else:
                self.data_matrix = np.vstack((self.data_matrix, curr_class_data_mat))

    def __len__(self):
        return self.data_matrix.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # return torch.from_numpy(self.data_matrix[idx, 0:2]), torch.tensor(self.data_matrix[idx][2].astype(int))
        return torch.FloatTensor(self.data_matrix[idx, 0 : self.dim]), torch.tensor(
            self.data_matrix[idx][-1]
        ).type(torch.int64)

    def get_all_data_as_numpy(self):
        return self.data_matrix


def load_data(input_size=2, num_classes=2, size_list=[100, 100]):
    """
    specify mean_list and cov_mat for each class,
    specify size of each class (in size_list)
    (dimension of data automatically sets to shape of mean_list)
    """
    ## new ideas experiments
    mean_lists = [[-4.0, -0.2]]
    cov_mats = [[[1.0, 0.0], [0.0, 1.0]]]

    mean_lists.append([-1.0, 3.5])
    cov_mats.append([[1.0, 0.0], [0.0, 1.0]])

    mean_lists.append([3.5, 1])
    cov_mats.append([[1, 0.5], [0.5, 3.5]])

    # mean_lists.append([0, 1])
    # cov_mats.append([[1, -0.5],
    #                  [-0.5, 3.5]])

    mean_lists = np.array(mean_lists)
    cov_mats = (np.array(cov_mats) * 5).tolist()
    size_list = [300, 300, 300]  # , 300

    # ##################3 original "khatti konande data" #
    # mean_lists = [[-4.0, -.2]]
    # cov_mats = [[[1.0, 0.0],
    #              [0.0, 1.0]]]

    # mean_lists.append([-1.0, 3.5])
    # cov_mats.append([[1.0, 0.0],
    #                  [0.0, 1.0]])

    # mean_lists.append([3.5, 1])
    # cov_mats.append([[1, 0.5],
    #                  [0.5, 3.5]])
    # mean_lists = np.array(mean_lists)
    # size_list = [250, 275, 300]

    #####  dade ha roo ham oftadan
    # mean_lists = [[-1.0, 1.]]
    # cov_mats = [[[1.0, 0.0],
    #              [0.0, 1.0]]]

    # mean_lists.append([.5, .5])
    # cov_mats.append([[3.0, 0.0],
    #                  [0.0, .5]])

    # mean_lists.append([-.5, -.5])
    # cov_mats.append([[1.5, -0.5],
    #                  [-0.5, 3.5]])

    # mean_lists = np.array(mean_lists)
    # size_list = [400, 400, 400]

    # # #####################zarbdari
    # mean_lists = [[0, -6.]]
    # cov_mats = [[[1.0, 0.0],
    #              [0.0, 1.0]]]

    # mean_lists.append([0.0, 0.0])
    # cov_mats.append([[3.0, 1.5],
    #                  [1.5, 1.0]])

    # mean_lists.append([0.0, 0.0])
    # cov_mats.append([[3, -1.5],
    #                  [-1.5, 1.]])

    # mean_lists = np.array(mean_lists)
    # size_list = [333, 333, 334]

    # mosalasi
    # mean_lists = [[2., 2.]]
    # cov_mats = [[[1., -1.0],
    #              [-1.0, 3.0]]]
    #
    # mean_lists.append([-2., 2.0])
    # cov_mats.append([[1., 1.],
    #                  [1., 3.0]])
    #
    # mean_lists.append([0.0, -0.5])
    # cov_mats.append([[3., 0.0],
    #                  [0.0, 1.]])
    #
    # mean_lists = np.array(mean_lists)
    # size_list = [300, 300, 300]

    # mean_lists = [[-4, 2]]
    # cov_mats   = [[[2.0, 0.0],
    #                 [0.0, 1.0]]]

    # mean_lists.append([-2, -2])
    # cov_mats.append([[1.0, 0.0],
    #                     [0.0, 2.0]])

    # mean_lists.append([-4, 4])
    # cov_mats.append([[2, 0.0],
    #                     [0.0, 3]])

    # 3d:
    # mean_lists = [[-4, 3, 6]]
    # cov_mats   = [[ [6.0, 0.0, 1.0],
    #                 [0.0, 4.0, 0.0],
    #                 [1.0, 0.0, 3]]]

    # mean_lists.append([-2, -2, 0])
    # cov_mats.append([ [4.0, 0.0, 0.7],
    #                 [0.0, 13, 0.1],
    #                 [0.7, 0.1, 7]])

    # mean_lists.append([0, 2, -2])
    # cov_mats.append([ [5.0, 0.4, 0.3],
    #                 [0.4, 4.0, 0.0],
    #                 [0.3, 0.0, 2.5]])

    # mean_lists.append([10, 10, 10])
    # cov_mats.append([ [5.0, 0.4, 0.3],
    #                 [0.4, 4.0, 0.0],
    #                 [0.3, 0.0, 2.5]])
    # size_list = [200, 300, 400, 100]

    num_classes = len(mean_lists)
    dataset_train = multi_var(
        mean_lists=mean_lists,
        cov_mats=cov_mats,
        size_list=size_list,
        num_classes=num_classes,
        np_random_seed=70,
    )
    dataset_test = multi_var(
        mean_lists=mean_lists,
        cov_mats=cov_mats,
        size_list=size_list,
        num_classes=num_classes,
        np_random_seed=1,
    )
    input_size = dataset_train.dim

    return input_size, num_classes, dataset_train, dataset_test, sum(size_list)


if __name__ == "__main__":
    test, _, _, _ = get_cifar10_data(11, True, 128)
