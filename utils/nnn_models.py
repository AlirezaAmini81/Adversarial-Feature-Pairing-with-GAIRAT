import torch
import torch.nn as nn
import numpy as np

from acc_exp.models import ModLenet

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(input_size, model_name, num_classes, feat_dim, feature_extract=False):
    model_ft = None
    if model_name == 'dorfer_mnist':
        model_ft = dorfer_mnist()
    elif model_name=='one_layer_lili':
       model_ft = one_layer_lili(input_size, feat_dim, num_classes)
    elif model_name=='four_layer_lili':
       model_ft = four_layer_lili(input_size, feat_dim, num_classes)
    elif model_name=='two_layer_ce':
       model_ft = two_layer_ce(input_size, feat_dim, num_classes)
    elif model_name=='hamedNet':
        model_ft = hamedNet(input_size, feat_dim, num_classes)
    elif model_name=='liliNet':
        model_ft = liliNet(input_size, feat_dim, num_classes)
    elif model_name=='lenetspp':
        model_ft = lenetspp(feat_dim, num_classes)
    elif model_name=='MNIST_VGG':
        model_ft = MNIST_VGG(num_classes)
    elif model_name=='twoLayerNet':
        model_ft = twoLayerNet(input_size, feat_dim, num_classes)
    elif model_name=='threeLayerNet':
        model_ft = threeLayerNet(input_size, feat_dim, num_classes)
    elif model_name=='lenetmm':
        model_ft = lenetsmm(feat_dim, num_classes)
    elif model_name == 'cifar2d':
        model_ft = CIFAR2d()
    elif model_name == 'ccl_mnist_2d':
        model_ft = ccl_mnist_2d(feat_dim)
    elif model_name == 'lenet': 
        model_ft = lenet()
    elif model_name == 'ModLenet': 
        model_ft = ModLenet()
    else:
        print("invalid model name.")
        exit()

    set_parameter_requires_grad(model_ft, feature_extract)
    return model_ft




class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
        self.modelName = self.__class__.__name__

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.prelu2 = nn.PReLU()
        self.fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.prelu3 = nn.PReLU()
        self.fc2 = nn.Linear(120, 2)
        self.prelu4 = nn.PReLU()
        self.softmax_weights = nn.Linear(2, 10, bias=False)
        
        is_valid_model = False
        for name, param in self.named_parameters():
            if 'softmax_weights' in name:
                is_valid_model = True
                break
        
        assert is_valid_model, 'model s last layer should be named `softmax_weights`'

    def forward(self, x):
        # print(x.shape)
        # print(x.device)
        x = torch.nn.functional.max_pool2d(self.prelu1(self.conv1(x)), (2, 2))
        x = torch.nn.functional.max_pool2d(self.prelu2(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = self.prelu3(self.fc1(x))
        feats = self.prelu4(self.fc2(x))
        logits = self.softmax_weights(feats)
        return feats, torch.nn.functional.log_softmax(logits, dim=1)
    
    def normalize_weights(self, alpha = 1.0):
        norms = torch.norm(self.softmax_weights.weight.data, p=2, dim=1)
        self.softmax_weights.weight.data = (self.softmax_weights.weight.data.T / norms).T * alpha 


class Tanh(nn.Module):
    def forward(self, x):
        return 1.7159 * torch.tanh(x * 2 / 3)
        # return torch.tanh(x)

class dorfer_mnist(nn.Module):
    """
    feat_dim -> dimensions of features that we want to extract as aux_output
    input_size -> input_size
    num_classes -> number of classes in dataset, some models
        (like centerloss mode) set the last layer number of neurons to
        the number of classes.
    """

    def __init__(self):
        super(dorfer_mnist, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 64, 3, stride = 1, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(64, 64, 3, stride = 1, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(64, 96, 3, stride = 1, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(96)
        self.relu2_1 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(96, 96, 3, stride = 1, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(96)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(96, 256, 3, stride = 1, padding=0)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU()

        self.conv4_1 = nn.Conv2d(256, 256, 1, stride = 1, padding=0)
        self.conv4_1_bn = nn.BatchNorm2d(256)
        self.relu4_1 = nn.ReLU()

        # classification layers:
        self.conv5_1 = nn.Conv2d(256, 10, 1, stride = 1, padding=0)
        self.conv5_1_bn = nn.BatchNorm2d(10)
        self.relu5_1 = nn.ReLU()

        self.pool5 = nn.AvgPool2d(5, count_include_pad=False)
        
        # initialize model weights
        self.apply(self.init_normal)
    
    def init_normal(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight)
    
    def forward(self, x):
        # original implementation used [Conv -> Relu -> BatchNorm], 
        # you are using [Conv -> batchNorm -> Relu]

        x = self.relu1_1(self.conv1_1_bn(self.conv1_1(x)))
        x = self.relu1_2(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(x)
        # x = nn.functional.dropout2d(x, 0.25)
        
        x = self.relu2_1(self.conv2_1_bn(self.conv2_1(x)))
        x = self.relu2_2(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(x)
        # x = nn.functional.dropout2d(x, 0.25)

        x = self.relu3_1(self.conv3_1_bn(self.conv3_1(x)))
        # x = nn.functional.dropout2d(x, 0.50)


        x = self.relu4_1(self.conv4_1_bn(self.conv4_1(x)))
        # x = nn.functional.dropout2d(x, 0.50)

        ### original implementation used non-linearity in the last layer
        # x = self.conv5_1_bn(self.relu5_1(self.conv5_1(x)))
        x = self.conv5_1_bn(self.conv5_1(x))
        x = self.pool5(x)
        
        x = torch.squeeze(x)
        
        # in the form 'return aux_outputs, outs'
        return x, x

    ### Original Implementation: 
    # def forward(self, x):
        
    #     x = self.conv1_1_bn(self.relu1_1(self.conv1_1(x)))
        
    #     x = self.conv1_2_bn(self.relu1_2(self.conv1_2(x)))
        
    #     x = self.pool1(x)
        
    #     x = self.conv2_1_bn(self.relu2_1(self.conv2_1(x)))
    #     x = self.conv2_2_bn(self.relu2_2(self.conv2_2(x)))
    #     x = self.pool2(x)
        
    #     x = self.conv3_1_bn(self.relu3_1(self.conv3_1(x)))
        
    #     x = self.conv4_1_bn(self.relu4_1(self.conv4_1(x)))
        
    #     ### original implementation used non-linearity in the last layer
    #     x = self.conv5_1_bn(self.relu5_1(self.conv5_1(x)))
    #     x = self.pool5(x)
        
    #     x = torch.squeeze(x)
        
    #     # in the form 'return aux_outputs, outs'
    #     return x, x

class one_layer_lili(nn.Module):
    """
    feat_dim -> dimensions of features that we want to extract as aux_output
    input_size -> input_size
    num_classes -> number of classes in dataset, some models
        (like centerloss mode) set the last layer number of neurons to
        the number of classes.
    """

    def __init__(self, input_size, feat_dim: int, num_classes: int):
        super(one_layer_lili, self).__init__()
        self.input_size = input_size
        self.feat_dim = feat_dim
        # self.num_classes = num_classes

        self.flatten = nn.Flatten()
        self.L1 = nn.Sequential(
            nn.Linear(self.input_size, self.feat_dim)
        )

        # initialize model weights
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.flatten(x)

        x_divergence1 = self.L1(x)

        # in the form 'return aux_outputs, outs'
        return x_divergence1, x_divergence1

    def _init_weights(self, module):
        """
        ^?
        """
        if self.input_size == self.feat_dim:
            if isinstance(module, nn.Linear):
                module.weight.data = torch.eye(self.feat_dim)


class four_layer_lili(nn.Module):
    """
    feat_dim -> dimensions of features that we want to extract as aux_output
    input_size -> input_set the last layer number of neurons to
        the number of classes.size
    num_classes -> number of classes in dataset, some models
        (like centerloss mode)
    """
    def __init__(self, input_size, feat_dim, num_classes):
        super(four_layer_lili, self).__init__()
        self.input_size = input_size
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.input_size, 350)
        self.act1 = nn.PReLU()

        self.fc2 = nn.Linear(350, 128)
        self.act2 = nn.PReLU()

        self.fc3 = nn.Linear(128, 50)
        self.act3 = nn.PReLU()

        self.fc4 = nn.Linear(50, self.feat_dim)

    def forward(self, x):
        x = self.flatten(x)

        z1 = self.fc1(x)
        a1 = self.act1(z1)

        z2 = self.fc2(a1)
        a2 = self.act2(z2)

        z3 = self.fc3(a2)
        a3 = self.act3(z3)

        z4 = self.fc4(a3)

        return z4, z4

class twoLayerNet(nn.Module):
    """
    feat_dim -> dimensions of features that we want to extract as aux_output
    input_size -> input_size
    num_classes -> number of classes in dataset, some models
        (like centerloss mode) set the last layer number of neurons to
        the number of classes.
    """

    def __init__(self, input_size, feat_dim, num_classes):
        super(twoLayerNet, self).__init__()
        self.input_size = input_size
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        dim1 = 20

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.input_size, dim1)
        # self.act1 = Tanh()
        self.act1 = nn.PReLU()

        self.fc2 = nn.Linear(dim1, self.feat_dim)

    def forward(self, x):
        x = self.flatten(x)

        z1 = self.fc1(x)
        a1 = self.act1(z1)

        z2 = self.fc2(a1)
        return z2, z2


class threeLayerNet(nn.Module):
    """
    feat_dim -> dimensions of features that we want to extract as aux_output
    input_size -> input_size
    num_classes -> number of classes in dataset, some models
        (like centerloss mode) set the last layer number of neurons to
        the number of classes.
    """

    def __init__(self, input_size, feat_dim, num_classes):
        super(threeLayerNet, self).__init__()
        self.input_size = input_size
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.flatten = nn.Flatten()

        dim1 = 20
        dim2 = 10
        
        self.fc1 = nn.Linear(self.input_size, dim1)
        # self.act1 = Tanh()
        self.act1 = nn.PReLU()
        
        self.fc2 = nn.Linear(dim1, dim2)
        # self.act2 = Tanh()
        self.act2 = nn.PReLU()

        self.fc3 = nn.Linear(dim2, self.feat_dim)

    def forward(self, x):
        x = self.flatten(x)

        z1 = self.fc1(x)
        a1 = self.act1(z1)

        z2 = self.fc2(a1)
        a2 = self.act2(z2)
        

        z2 = self.fc3(a2)
        return z2, z2


class two_layer_ce(nn.Module):
    """
    feat_dim    -> dimensions of features that we want to extract as aux_output
    input_size  -> input_size
    num_classes -> number of classes in dataset, some models
        (like centerloss mode) set the last layer number of neurons to
        the number of classes.
    """

    def __init__(self, input_size, feat_dim, num_classes):
        super(two_layer_ce, self).__init__()
        self.input_size = input_size
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.input_size, self.feat_dim)
        self.fc2 = nn.Linear(self.feat_dim, self.num_classes)
        self.prelu1 = nn.PReLU()

    def forward(self, x):
        x = self.flatten(x)
        z1 = self.fc1(x)
        a1 = self.prelu1(z1)
        z2 = self.fc2(a1)

        z2_softmax_log = torch.nn.functional.log_softmax(z2, dim=1)  # (num_classes, 1)

        return z1, z2_softmax_log


# hamedNet
class hamedNet(nn.Module):
    """
    feat_dim -> dimensions of features that we want to extract as aux_output
    input_size -> input_size
    num_classes -> number of classes in dataset, some models
        (like centerloss mode) set the last layer number of neurons to
        the number of classes.
    """

    def __init__(self, input_size, feat_dim, num_classes):
        super(hamedNet, self).__init__()
        self.input_size = input_size
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.input_size, 350)
        self.act1 = nn.PReLU()

        self.fc2 = nn.Linear(350, 128)
        self.act2 = nn.PReLU()

        self.fc3 = nn.Linear(128, 50)
        self.act3 = nn.PReLU()

        self.fc4 = nn.Linear(50, self.feat_dim)
        self.act4 = nn.PReLU()

        self.fc5 = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        x = self.flatten(x)

        z1 = self.fc1(x)
        a1 = self.act1(z1)

        z2 = self.fc2(a1)
        a2 = self.act2(z2)

        z3 = self.fc3(a2)
        a3 = self.act3(z3)

        z4 = self.fc4(a3)
        a4 = self.act4(z4)

        z5 = self.fc5(a4)
        z5_softmax_log = torch.nn.functional.log_softmax(z5, dim=1)

        return z4, z5_softmax_log


def weight_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out)).item()
        m.weight.data.normal_(0.0, variance)

class liliNet(nn.Module):
    def __init__(self, input_size, feat_dim, num_classes):
        super(liliNet, self).__init__()
        self.input_size = input_size
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        dim1 = 1500
        dim2 = 2500
        dim3 = 100

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, dim1)
        self.fc_bn1 = nn.BatchNorm1d(dim1)
        self.fc2 = nn.Linear(dim1, dim2)
        self.fc_bn2 = nn.BatchNorm1d(dim2)
        self.fc3 = nn.Linear(dim2, dim3)
        self.fc_bn3 = nn.BatchNorm1d(dim3)
        self.fc4 = nn.Linear(dim3, feat_dim)
        weight_init(self.fc1)
        weight_init(self.fc2)
        weight_init(self.fc3)
        weight_init(self.fc4)

        self.activate1 = nn.ReLU()
        self.activate2 = nn.ReLU()
        self.activate3 = nn.ReLU()
        # self.activate1 = nn.Tanh()
        # self.activate2 = nn.Tanh()
        # self.activate3 = nn.Tanh()

    def forward(self, x):
        
        x = self.fc1(self.flatten(x))
        x = self.activate1(x)
        x = self.fc_bn1(x)
        x = self.fc2(x)
        x = self.activate2(x)
        # x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc_bn2(x)
        x = self.fc3(x)
        x = self.activate3(x)
        x = self.fc_bn3(x)

        x = self.fc4(x)

        return x, x



class lenetsmm(nn.Module):
    """
    model used for 2d visualization of MNIST on CCL and CL paper and large softmax ... paper.
    goes to 2D space right before handing over the features to the softmax layer in the end. 
    """
    # remove batch norm if didnt work
    def __init__(self, feat_dim, num_classes) -> None:
        super(lenetsmm, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        # figure size + 1 - kernel size
        self.conv1_1 =  nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, padding=2), 
                                      nn.BatchNorm2d(32)
                                      )
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 =  nn.Sequential(nn.Conv2d(32, 32, kernel_size=5, padding=2), 
                                      nn.BatchNorm2d(32)
                                      )
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 =  nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, padding=2), 
                                      nn.BatchNorm2d(64)
                                      )
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 =  nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, padding=2), 
                                      nn.BatchNorm2d(64)
                                      )
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 =  nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, padding=2), 
                                      nn.BatchNorm2d(128)
                                      )
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 =  nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2), 
                                      nn.BatchNorm2d(128)
                                      )
        self.prelu3_2 = nn.PReLU()
        self.ip1 = nn.Linear(128 * 3 * 3, self.feat_dim)
        # self.preluip1 = nn.PReLU()
        
        # self.ip2 = nn.Linear(feat_dim, num_classes, bias= False) # should bias be false? 

    def forward(self, x): 
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)
        ip1_before_relu = self.ip1(x)
        
        # ip1_after_relu = self.preluip1(ip1_before_relu)

        # ip2 = self.ip2(ip1_after_relu)

        # in the form 'return aux_outputs, outs'
        # shoud return ip1_before_relu or ip1_after_relu as aux output? 
        # its important since loss is applied to it
        
        # return ip1_before_relu, nn.functional.log_softmax(ip2, dim=1)
        return ip1_before_relu, ip1_before_relu


class lenetspp(nn.Module):
    """
    model used for 2d visualization of MNIST on CCL and CL paper and large softmax ... paper.
    goes to 2D space right before handing over the features to the softmax layer in the end. 
    """
    # remove batch norm if didnt work
    def __init__(self, feat_dim, num_classes) -> None:
        super(lenetspp, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        # figure size + 1 - kernel size
        self.conv1_1 =  nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, padding=2), 
                                      # nn.BatchNorm2d(32)
                                      )
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 =  nn.Sequential(nn.Conv2d(32, 32, kernel_size=5, padding=2), 
                                      # nn.BatchNorm2d(32)
                                      )
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 =  nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, padding=2), 
                                      # nn.BatchNorm2d(64)
                                      )
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 =  nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, padding=2), 
                                      # nn.BatchNorm2d(64)
                                      )
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 =  nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, padding=2), 
                                      # nn.BatchNorm2d(128)
                                      )
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 =  nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2), 
                                      # nn.BatchNorm2d(128)
                                      )
        self.prelu3_2 = nn.PReLU()
        self.ip1 = nn.Linear(128 * 3 * 3, self.feat_dim)
        self.preluip1 = nn.PReLU()
        
        self.ip2 = nn.Linear(feat_dim, num_classes, bias= False) # should bias be false? 

    def forward(self, x): 
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)
        ip1_before_relu = self.ip1(x)
        
        ip1_after_relu = self.preluip1(ip1_before_relu)

        ip2 = self.ip2(ip1_after_relu)

        # in the form 'return aux_outputs, outs'
        # shoud return ip1_before_relu or ip1_after_relu as aux output? 
        # its important since loss is applied to it
        
        # return ip1_before_relu, nn.functional.log_softmax(ip2, dim=1)
        return ip1_after_relu, nn.functional.log_softmax(ip2, dim=1)


class MNIST_VGG(nn.Module): 
    # bs = 256
    # weight decay = 0.0005
    # momentum = 0.9
    def __init__(self, num_classes):
        super(MNIST_VGG, self).__init__()
        self.num_classes = num_classes

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.pool3 = nn.MaxPool2d(2, 2) # 64 * 3 * 3
        
        self.FC1 = nn.Sequential(
            nn.Linear(64*3*3, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
        )
        
        self.FC2 = nn.Sequential(
            # should i put BN and non-linearity in last layer? 
            nn.Linear(256, self.num_classes)
            # nn.BatchNorm1d(256)
            # nn.PReLU()
        )

    def forward(self, x): 
        # x -> 1, 28, 28
        x = self.conv0(x) # 64, 28, 28
        x = self.conv1(x) # 64, 28, 28
        x = self.pool1(x) # 64, 14, 14
        x = self.conv2(x) # 64, 14, 14
        x = self.pool2(x) # 64, 7, 7
        x = self.conv3(x) # 64, 7, 7
        x = self.pool3(x) # 64, 3, 3
        x = x.view(-1, 64 * 3 * 3) # 576
        ip1 = self.FC1(x)   # 256
        ip2 = self.FC2(ip1)   # 10

        # format: return aux_out, out
        return ip1, nn.functional.log_softmax(ip2, dim=1) 
        


class CIFAR2d(nn.Module):
    def __init__(self) -> None:
        super(CIFAR2d, self).__init__()
        
        # Conv0.x
        self.conv0x = nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64)),
            nn.PReLU(),
        )
        
        # Conv1.x
        self.conv1x = nn.Sequential(
            nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)),
            nn.PReLU(),
            nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)),
            nn.PReLU(),
            nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)),
            nn.PReLU(),
            nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)),
            nn.PReLU(),
        )
        
        # Pool1
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv2.x
        self.conv2x = nn.Sequential(
            nn.Sequential(nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96)),
            nn.PReLU(),
            nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96)),
            nn.PReLU(),
            nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96)),
            nn.PReLU(),
            nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96)),
            nn.PReLU(),
        )
        
        # Pool2
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3x = nn.Sequential(
            nn.Sequential(nn.Conv2d(96, 128, 3, padding=1), nn.BatchNorm2d(128)),
            nn.PReLU(),
            nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128)),
            nn.PReLU(),
            nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128)),
            nn.PReLU(),
            nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128)),
            nn.PReLU(),
        )
        
        # Pool3
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # flatten [128, 4, 4] to 2048
        self.flatten = nn.Flatten()

        # fully connected 
        self.FC1 = nn.Sequential(
            nn.Linear(128*4*4, 2),
            nn.BatchNorm1d(2),
            nn.PReLU(),
        )

        self.FC2 = nn.Sequential(
            nn.Linear(2, 10)
        )
        

    def forward(self, x):
        # print(x.shape)
        x = self.conv0x(x)
        # print(x.shape)
        x = self.conv1x(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2x(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.conv3x(x)
        # print(x.shape)
        x = self.pool3(x)
        # print(x.shape)

        x = self.flatten(x)
        ip1 = self.FC1(x)
        ip2 = self.FC2(ip1)

        return ip1, nn.functional.log_softmax(ip2, dim=1)
    

class ccl_mnist_2d(nn.Module):
    def __init__(self, feat_dim):
        super(ccl_mnist_2d, self).__init__()
        self.feat_dim = feat_dim
        # figure size + 1 - kernel size
        self.conv1_1 =  nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32))
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 =  nn.Sequential(nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32))
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 =  nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.BatchNorm2d(64))
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 =  nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, padding=2), nn.BatchNorm2d(64))
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 =  nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, padding=2), nn.BatchNorm2d(128))
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 =  nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2), nn.BatchNorm2d(128))
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128 * 3 * 3, self.feat_dim)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)
        ip1 = self.preluip1(self.ip1(x))
        return ip1