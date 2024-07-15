import torch
import numpy as np

def get_modified_conf_mat(conf_mat: torch.Tensor) -> torch.Tensor:
    ###  Old code: 
    ## select classes i, j based on confusion matrix: 
    # confusion_mat.fill_diagonal_(-1) # don't fill with 0, 
    # # there might be some batch that is predicted 100p accurate,
    # # and i, j might be selected in diagonal which is wrong?  
    # i, j = (confusion_mat==torch.max(confusion_mat)).nonzero()[0] # if multiple maximum, pick first one
    # print('conf i, j: ', i.item(), j.item())

    ## confusion_mat.sum(1): sum along rows of conf_mat
    # weights = torch.nn.functional.softmax(confusion_mat.sum(1), dim=0).to(features.device)
    # print('-----')
    # print(conf_mat)
    # print(conf_mat.sum(1))
    # print(conf_mat/conf_mat.sum(1))
    # print((conf_mat/conf_mat.sum(1).unsqueeze(1)).sum(1))
    
    conf_mat = conf_mat / conf_mat.sum(1).unsqueeze(1)
    # print('-----')
    
    conf_mat.fill_diagonal_(0)
    conf_mat = torch.tril(conf_mat) + torch.triu(conf_mat).T
    conf_mat += conf_mat.clone().T

    return conf_mat

def get_scatters(aux_outputs, labels, 
                 num_classes, weighted=True, debug=False):
    """
    get_scatters returns st_list, sw_list and sb_list
    st_list -- [d, ],  st_list[i] is variance of aux_outputs over dimension "i"
    sw_list -- [c, d], sw_list[i, j] is the weighted(w=relative freq) variance of aux_outputs with label "i" over dimension "j"
    sb_list -- [d, ],  sb_list[i] is

    Args:
    aux_outputs -- input data to Sw, Sb, St
    labels -- labels of aux_outputs, used to split data into C classes
    num_classes -- number of classes
    """

    Ni = torch.bincount(labels, minlength=num_classes)  # size of each class in aux_outputs
    N = aux_outputs.shape[0]  # size of aux_outputs

    overall_mean = torch.mean(aux_outputs, axis=0)  # [d, ], mean of all aux_outputs (separately over each dimension)
    mean_classes = []  # [c, d], mean_classes[i, j] -> within mean of class "i" over dimension "j"
    for c in range(num_classes):
        # curr_class_mean = torch.nan_to_num(torch.mean(aux_outputs[torch.where(labels == c)], axis=0))  # [d, ]
        curr_class_mean = torch.mean(aux_outputs[torch.where(labels == c)], axis=0)  # [d, ]
        
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
    st_list = (1 / (N)) * ((aux_outputs - overall_mean) * (aux_outputs - overall_mean)).sum(dim=0)  # [d, ]

    """ Sb: 
    ^?
    """
    mean_dist_classes = []  # [c, d] , arr[class c]: weighted distance of mean of class c with overall mean
    for curr_c in range(num_classes):
        temp = (mean_classes[curr_c] - overall_mean) * (mean_classes[curr_c] - overall_mean)  # [d, ]

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
        # curr_class_data = torch.nan_to_num(aux_outputs[torch.where(labels == curr_c)])  # datas with label = curr_c
        curr_class_data = aux_outputs[torch.where(labels == curr_c)]  # datas with label = curr_c

        if curr_class_data.shape[0] == 0:
            print(labels)
            print(f'current batch does not include class {curr_c}.')
            # exit()

        temp = (curr_class_data - mean_classes[curr_c]) * \
               (curr_class_data - mean_classes[curr_c])
        temp = (1 / Ni[curr_c]) * temp.sum(dim=0)  # [d, ] , mean over all curr_class_data

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


    return st_list, sw_list, sb_list


def get_covs(aux_outputs, labels, num_classes):
    '''
    aux_outputs: [m, dim] , m is number of samples, dim is dimension of features(number of variables)
    '''
    # assert False, 'S_b is wrong, when you apply weights, you have to compute them manually not use troch.cov, read Dr.Ghiasi paper for more details'
    # assert sb_weighted==True, 'false is wrong mathimatically'
    # return None 
    Ni = torch.bincount(labels, minlength=num_classes)  # [c, ]
    N = aux_outputs.shape[0]  # [1, ]

    
    # St    
    cov_t = torch.cov(aux_outputs.T)  # [d, d]

    # Sw
    covs_w = []  # [c, d, d]
    for c in range(num_classes):
        curr_class_data = aux_outputs[torch.where(labels == c)]
        temp = torch.cov(curr_class_data.T)
        covs_w.append(temp)
    covs_w = torch.stack(covs_w)

    # Sb
    mu_t = torch.mean(aux_outputs, axis=0).unsqueeze(1)  # [d, 1]
    mu_c = [torch.nan_to_num(torch.mean(aux_outputs[torch.where(labels == i)], axis=0)) for i in
                    range(num_classes)]  # [c, d]
    mu_c = torch.stack(mu_c) # [c, d]
    # print(mu_c)
    cov_b = torch.cov(mu_c.T)  # [d, d]

    return cov_t, covs_w, cov_b

def get_covs_unbalanced(aux_outputs, labels, num_classes):
    '''
    aux_outputs: [m, dim] , m is number of samples, dim is dimension of features(number of variables)
    '''
    Ni = torch.bincount(labels, minlength=num_classes)  # [c, ]
    N = aux_outputs.shape[0]  # [1, ]

    # St    
    cov_t = torch.cov(aux_outputs.T)  # [d, d]

    # Sw
    covs_w = []  # [c, d, d]
    for c in range(num_classes):
        curr_class_data = aux_outputs[torch.where(labels == c)]
        temp = torch.cov(curr_class_data.T)
        covs_w.append(temp)
    covs_w = torch.stack(covs_w)

    # Sb
    mu_t = torch.mean(aux_outputs, axis=0).unsqueeze(1)  # [d, 1]
    mu_c = [torch.nan_to_num(torch.mean(aux_outputs[torch.where(labels == i)], axis=0)) for i in
                    range(num_classes)]  # [c, d]
    
    
    sbs = []
    # Dr ghiasi paper, page 3 
    for c in range(len(mu_c)): 
        sbs.append(
            torch.mm(mu_c[c].unsqueeze(1)-mu_t, (mu_c[c].unsqueeze(1)-mu_t).T) 
        )
    cov_b =  (1/(num_classes-1)) * torch.stack(sbs).sum(0) # 

    return cov_t, covs_w, cov_b


def get_dist_centroids(aux_outputs, labels, num_classes, weighted=True):
    '''
    get_dist_centroids returns pairwise "distance" of classes.
    "distance" is calculated as follows:
        d[class_i, class_j] = (euclidean distance of mean_i and mean_j)^2

    Note that means of classes ARE weighted.
    (they ARE multiplied by their respective relative frequency).

    Args:
    aux_outputs -- input data to compute variances for
    labels -- labels of aux_outputs, used to split data into C classes
    num_classes -- number of classes
    weighted -- if true, means are multiplied by their relative frequency

    Returned Args:
    dist_centroids[num_classes, num_classes] --> dist_centroid[i, j] = "distance" of class i and j
    (distance of a class with it self is set to float("Inf"))
    '''
    mean_classes = [(torch.mean(aux_outputs[torch.where(labels == i)], axis=0)) for i in range(num_classes)]
    mean_classes = torch.stack(mean_classes)

    Ni = torch.bincount(labels, minlength=num_classes)
    N = aux_outputs.shape[0]

    weighted_dist_centroids = torch.zeros(num_classes, num_classes)

    for class_i in range(num_classes):
        for class_j in range(num_classes):
            if class_i == class_j:
                # set to inf
                weighted_dist_centroids[class_i, class_j] = float("Inf")
            elif class_i < class_j:
                if weighted == True:
                    temp = ((Ni[class_i] / N) * mean_classes[class_i] - (Ni[class_j] / N) * mean_classes[
                        class_j])  # [d, ]
                else:
                    # to get same results as def get_vect_dist_centroids and lili's j1
                    # remove the multiplication by relative frequenct
                    temp = (mean_classes[class_i] - mean_classes[class_j])  # [d, ]

                temp = temp * temp  # [d, ]
                temp = temp.sum()  # sum over dimensions
                weighted_dist_centroids[class_i, class_j] = temp

            elif class_i > class_j:
                # since weighted_dist_centroids is a symmetric matrix:
                weighted_dist_centroids[class_i, class_j] = weighted_dist_centroids[class_j, class_i]

    return weighted_dist_centroids


def get_vect_dist_centroids(aux_outputs, labels, num_classes):
    '''
    get_vect_dist_centroids is the vectorized version of get_dist_centroids, (using LiLi's implementation)
    it returns pairwise "distance" of classes.
    "distance" is calculated as follows:
        d[class_i, class_j] = (euclidean distance of mean_i and mean_j)^2

    Note that means of classes are NOT weighted.
    (they are NOT multiplied by their respective relative frequency).

    Args:
    aux_outputs -- input data to compute variances for
    labels -- labels of aux_outputs, used to split data into C classes
    num_classes -- number of classes

    Returned Args:
    dist_centroids[num_classes, num_classes] -- dist_centroid[i, j] = "distance" of class i and j
    (distance of a class with it self is set to float("Inf"))
    '''

    mean_classes = [(torch.mean(aux_outputs[torch.where(labels == i)], axis=0)) for i in range(num_classes)]
    mean_classes = torch.stack(mean_classes)

    dist_centroids = torch.norm(mean_classes[:, None] - mean_classes, dim=2,
                                p=2)  # [c, c], symmetric. This is the trick part of computing the pairewise distance between the centroids of classes.
    dist_centroids = torch.pow(dist_centroids, 2)  # [c, c], symmetric. Norm 2 squared

    dist_centroids.fill_diagonal_(float("Inf"))

    return dist_centroids



###################### Functions for Testing 



def get_c_var(aux_outputs, labels, num_classes):
    """
    get_c_var returns variance of each class (within) 
    and cross covariance of each class (cross covariance only applicable to 2d for now).

    Args:
    aux_outputs -- input data to compute variances for
    labels -- labels of aux_outputs, used to split data into C classes
    num_classes -- number of classes

    returns:
    c_var -- #[c, d]
    cross_cov -- #[c, ]
    """

    Ni = np.bincount(labels, minlength=num_classes)  # size of each class in aux_outputs
    N = aux_outputs.shape[0]  # size of aux_outputs

    c_var = []  # [c, d] -> c_var[i, j]: within variance of j-th dimension for class "i"
    cross_cov = []  # [c, ] -> only for 2d data

    for curr_c in range(num_classes):
        curr_class_data = aux_outputs[np.where(labels == curr_c)]  # datas with label = curr_c
        # curr_class_var = np.var(curr_class_data,  axis=0)

        curr_cov_mat = np.cov(curr_class_data.T, bias=True)  # compute covariance matrix for current class
        # c_var.append(curr_class_var)
        c_var.append(curr_cov_mat.diagonal())  # diagonal elements of covarinace matrix are variances

        ########### covariance between all variabels only availabel for 2d data
        cross_cov.append(curr_cov_mat[0][1])

    c_var = np.stack(c_var)  # [c, d]
    cross_cov = np.array(cross_cov)  # [c, ]

    return c_var, cross_cov


def test_dist_centroids(aux_outputs, labels, num_classes, debug=False):
    '''
    test_dist_centroids is for testing validity of "get_dist_centroids" and "get_vect_dist_centroids" methods
    '''
    st_list0, sw_list0, sb_list0 = get_scatters(aux_outputs, labels, num_classes)
    # st [d, ]
    # sw_list [c, d]
    # sb_list [d, ]

    dist1 = get_vect_dist_centroids(aux_outputs, labels, num_classes)
    dist2 = get_dist_centroids(aux_outputs, labels, num_classes, weighted=False)

    if debug:
        print("sum of differences: {:.4f}".format(torch.nan_to_num(dist1 - dist2).sum().item()))

    # st, sw, sb unused, only report as result
    st = st_list0.sum()  # sum over dimensions
    sb = sb_list0.sum()  # sum over dimensions
    sw = sw_list0.sum()  # sum over dimensions and classes

    # dummy loss 
    loss = st / sb

    results = {'loss': loss,
               'sw': sw,
               'sb': sb,
               'st': st,
               }
    return results



def dummy_loss_test_scatters(aux_outputs, labels, num_classes):
    '''
    test function to check validity of new "get_scatters" matrix with the old way of computing st, sb, sw
    '''
    Ni = torch.bincount(labels, minlength=num_classes)  # [c, ]
    N = aux_outputs.shape[0]  # [1, ]

    overall_mean = torch.mean(aux_outputs, axis=0)  # [d, ]
    mean_classes = [torch.nan_to_num(torch.mean(aux_outputs[torch.where(labels == i)], axis=0)) for i in
                    range(num_classes)]  # [c, d]

    sb_list = [Ni[i] / N * torch.dot(mean_classes[i] - overall_mean, mean_classes[i] - overall_mean) for i in
               range(num_classes)]  # [c, ]
    sb = torch.stack(sb_list, dim=0).sum()  # [1, ]

    st = (1 / N * (aux_outputs - overall_mean) * (aux_outputs - overall_mean)).sum(dim=0).sum()

    sw_list = [1 / N * ((aux_outputs[torch.where(labels == i)] - mean_classes[i]) * (
            aux_outputs[torch.where(labels == i)] - mean_classes[i])) for i in range(num_classes)]
    sw = torch.vstack(sw_list).sum(dim=1).sum()

    print('older version: ')
    print('sw:    ', sw.item())
    print('sb:    ', sb.item())
    print('sw+sb: ', (sw + sb).item())
    print('st:    ', st.item())

    st_list2, sw_list2, sb_list2 = get_scatters(aux_outputs, labels, num_classes)
    print('new version(get_scatters): ')
    print('sw:    ', sw_list2.sum().item())
    print('sb:    ', sb_list2.sum().item())
    print('sw+sb: ', (sw_list2.sum() + sb_list2.sum()).item())
    print('st:    ', st_list2.sum().item())

    # return dummy loss and dummy st, sb, sw
    results = {'loss': overall_mean.sum(),
               'sw': torch.tensor(0.0),
               'sb': torch.tensor(0.0),
               'st': torch.tensor(0.0),
               }
    return results