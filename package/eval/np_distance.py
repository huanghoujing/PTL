"""Numpy version of euclidean distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""
import numpy as np


def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, dist_type='cosine', cos_to_normalize=True):
    """Compute the euclidean or cosine distance of all pairs.
    Args:
        array1: numpy array with shape [m1, n]
        array2: numpy array with shape [m2, n]
        dist_type: one of ['cosine', 'euclidean']
    Returns:
        dist: numpy array with shape [m1, m2]
    """
    if dist_type == 'cosine':
        if cos_to_normalize:
            array1 = normalize(array1, axis=1)
            array2 = normalize(array2, axis=1)
        dist = - np.matmul(array1, array2.T)
        # Turn distance into positive value
        dist += 1
    elif dist_type == 'euclidean':
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        dist[dist < 0] = 0
        # Print('Debug why there is warning in np.sqrt')
        # np.seterr(all='raise')
        # for x in dist.flatten():
        #     try:
        #         np.sqrt(x)
        #     except:
        #         print(x)
        # Setting `out=dist` saves 1x memory size of `dist`
        np.sqrt(dist, out=dist)
    else:
        raise NotImplementedError
    return dist
