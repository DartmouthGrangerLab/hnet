# Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
# Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
# If you use this code, cite ___.
import numpy as np
import scipy as sp
import scipy.sparse


def dataset(is_trn, n_per_class):
    """loads the mnist dataset"""
    # INPUTS
    #   is_trn      - scalar (bool)
    #   n_per_class - scalar (int64) if -1, will use as many as possible
    # RETURNS
    #   data   - n_pts x n_pixels (bool)
    #   labels - n_pts x 1 (uint8)
    seed = 10 # for consistent rng results

    data = scipy.sparse.load_npz('../datasets/img_captchas/mnist_784.npz') # load mnist data
    img_sz = [28,28] # MUST be square for now
    image_data = data[:,:-1]
    label_data = np.array(data[:,784].todense())
    if is_trn: # load trn data
        data = image_data[0:60000]
        labels = label_data[0:60000]
    else: # load tst data
        data = image_data[60000:]
        labels = label_data[60000:]

    if np.min(data) != np.max(data): # normalize to range 0 --> 1
        data = data - np.min(data)
        data = data / np.max(data)

    data = scipy.sparse.csr_matrix(data > 0.5, dtype=np.uint8) # threshold

    # subset (was most of get_mnist_features)
    # uniq_labels = [0] + list(np.unique(labels).ravel()) # digits 0 --> 9
    uniq_labels = list(np.unique(labels)) # digits 0 --> 9
    n_classes = len(uniq_labels) # should be 10; digits 0 --> 9

    if n_per_class == -1: # use almost all (preserving equal N)
        label_counts = np.zeros(n_classes)
        for label in uniq_labels:
            label_counts[label] = np.sum(labels.ravel() == label)
        n_per_class = int(np.floor(np.min(label_counts)))

    class_data = []
    class_labels = []
    for label in uniq_labels:
        idx = np.argwhere(labels.ravel() == label).squeeze()
        curr_data = data[idx]
        curr_labels = labels[idx]

        if is_trn:
            data_mean = (curr_data.mean(axis=0) > 0.5) * 1
            data_diff = curr_data - data_mean
            data_rms = np.sqrt(np.power(data_diff, 2).mean(axis=1)) # sqrt should be unnecessary
            idx = np.array(np.argsort(data_rms, axis=0)).ravel() # sort, most prototypical first
        else: # tst
            rng = np.random.default_rng(seed)
            idx = rng.permutation(curr_data.shape[0])

        class_data.append(curr_data[idx[0:n_per_class]])
        class_labels.append(curr_labels[idx[0:n_per_class]])

    data = scipy.sparse.vstack([x for x in class_data]).todense()
    label_idx = np.array(np.vstack([x for x in class_labels])).ravel()
    return {'data':data, 'label_idx':label_idx}