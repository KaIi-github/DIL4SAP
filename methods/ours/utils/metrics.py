import ot
import scipy.io
import scipy.ndimage
import numpy as np
from PIL import Image
from numpy import random
from skimage.transform import resize


def normalize(x, method='standard', axis=None):
    '''Normalizes the input with specified method.
    Parameters
    ----------
    x : array-like
    method : string, optional
        Valid values for method are:
        - 'standard': mean=0, std=1
        - 'range': min=0, max=1
        - 'sum': sum=1
    axis : int, optional
        Axis perpendicular to which array is sliced and normalized.
        If None, array is flattened and normalized.
    Returns
    -------
    res : numpy.ndarray
        Normalized array.
    '''
    # TODO: Prevent divided by zero if the map is flat
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res

def calc_cc(gtsAnn, resAnn):
    """
    Computer CC score. A simple implementation
    :param gtsAnn : ground-truth fixation map
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]

def calc_nss(gtsAnn, resAnn):
    """
    Computer NSS score.
    :param gtsAnn : ground-truth annotations
    :param resAnn : predicted saliency map
    :return score: int : NSS score
    """

    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.sum(salMap*gtsAnn)/np.sum(gtsAnn)

def calc_auc(gtsAnn, resAnn, stepSize=.01, Nrand=100000):
    """
    Computer AUC score.
    :param gtsAnn : ground-truth annotations
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    salMap = resAnn - np.min(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.max(salMap)

    S = salMap.reshape(-1)
    # Sth = np.asarray([ salMap[y-1][x-1] for y,x in gtsAnn ])
    y, x = np.where(gtsAnn == 1)
    tmp = []
    for i in range(x.shape[0]):
        tmp.append(salMap[y[i], x[i]])
    Sth = np.array(tmp)

    Nfixations = len(x)
    Npixels = len(S)

    # sal map values at random locations
    randfix = S[np.random.randint(Npixels, size=Nrand)]

    allthreshes = np.arange(0, np.max(np.concatenate((Sth, randfix), axis=0)), stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes) + 2)
    fp = np.zeros(len(allthreshes) + 2)
    tp[-1] = 1.0
    fp[-1] = 1.0
    tp[1:-1] = [float(np.sum(Sth >= thresh)) / Nfixations for thresh in allthreshes]
    fp[1:-1] = [float(np.sum(randfix >= thresh)) / Nrand for thresh in allthreshes]

    auc = np.trapz(tp, fp)
    return auc

def calc_auc_salicon(gtsAnn, resAnn, stepSize=.01, Nrand=100000):
    """
    Computer AUC score of salicon dataset.
    :param gtsAnn : ground-truth annotations
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    salMap = resAnn - np.min(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.max(salMap)

    S = salMap.reshape(-1)
    # Sth = np.asarray([salMap[y-1][x-1] for y, x in gtsAnn])
    y, x = np.where(gtsAnn == 1)

    tmp = []
    for i in range(x.shape[0]):
        tmp.append(salMap[y[i], x[i]])
    Sth = np.array(tmp)
    Nfixations = len(gtsAnn)
    Npixels = len(S)

    # sal map values at random locations
    randfix = S[np.random.randint(Npixels, size=Nrand)]

    allthreshes = np.arange(0, np.max(np.concatenate((Sth, randfix), axis=0)), stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes) + 2)
    fp = np.zeros(len(allthreshes) + 2)
    tp[-1] = 1.0
    fp[-1] = 1.0
    tp[1:-1] = [float(np.sum(Sth >= thresh)) / Nfixations for thresh in allthreshes]
    fp[1:-1] = [float(np.sum(randfix >= thresh)) / Nrand for thresh in allthreshes]

    auc = np.trapz(tp, fp)
    return auc

def calc_auc_borji(fixation_map, saliency_map, n_rep=100, step_size=0.1, rand_sampler=None):
    """
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
    ROC curve created by sweeping through threshold values at fixed step size
    until the maximum saliency map value.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at random locations to the total number of random locations
    (as many random locations as fixations, sampled uniformly from fixation_map ALL IMAGE PIXELS),
    averaging over n_rep number of selections of random locations.
    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
        Human fixation map.
    n_rep : int, optional
        Number of repeats for random sampling of non-fixated locations.
    step_size : int, optional
        Step size for sweeping through saliency map.
    rand_sampler : callable
    S_rand = rand_sampler(S, F, n_rep, n_fix)
    Sample the saliency map at random locations to estimate false positive.
        Return the sampled saliency values, S_rand.shape=(n_fix,n_rep)
    Returns
    -------
    AUC : float, between [0,1]
    """
    if not isinstance(fixation_map, np.ndarray):
        print('End of Epoch')
        return 0.0
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) >0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        print(np.nonzero(fixation_map.ravel())[0])
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3)
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # For each fixation, sample n_rep values from anywhere on the saliency map
    if rand_sampler is None:
        r = random.randint(0, n_pixels, [n_fix, n_rep])
        S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
    else:
        S_rand = rand_sampler(S, F, n_rep, n_fix)
    # Calculate AUC per random split (set of random locations)
    auc = np.zeros(n_rep) * np.nan

    all_TP = []
    all_FP = []
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds)+2)
        fp = np.zeros(len(thresholds)+2)
        tp[0] = 0; tp[-1] = 1
        fp[0] = 0; fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
        all_TP.append(tp)
        all_FP.append(fp)
    return np.mean(auc), (all_TP, all_FP) # Average across random splits

def calc_auc_judd(saliency_map, fixation_map, jitter=True):
    '''
    AUC stands for Area Under ROC Curve.
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
    ROC curve is created by sweeping through threshold values
    determined by range of saliency map values at fixation locations.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at all other locations to the total number of possible other locations (non-fixated image pixels).
    AUC=0.5 is chance level.
    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
        Human fixation map.
    jitter : boolean, optional
        If True (default), a small random number would be added to each pixel of the saliency map.
        Jitter saliency maps that come from saliency models that have a lot of zero values.
        If the saliency map is made with a Gaussian then it does not need to be jittered
        as the values vary and there is not a large patch of the same value.
        In fact, jittering breaks the ordering in the small values!
    Returns
    -------
    AUC : float, between [0,1]
    '''
    epsilon = 1e-7
    try:
        saliency_map = np.array(saliency_map, copy=False)
        fixation_map = np.array(fixation_map, copy=False) > 0.5
    except:
        saliency_map = np.array(saliency_map.numpy(), copy=False)
        fixation_map = np.array(fixation_map.numpy(), copy=False) > 0.5

    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3)
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')
    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
        tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix + epsilon) # Ratio other saliency map values above threshold
    return np.trapz(tp, fp) # y, x

def cacl_sim(saliency_map1, saliency_map2):
    """
    sim between two different saliency maps when viewed as distributions
    (SIM=1 means the distributions are identical).
    This similarity measure is also called **histogram intersection**.
    Parameters
    ----------
    saliency_map1 : real-valued matrix
        If the two maps are different in shape, saliency_map1 will be resized to match saliency_map2.
    saliency_map2 : real-valued matrix
    Returns
    -------
    SIM : float, between [0,1]
    """
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3) # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have values between [0,1] and sum up to 1
    map1 = normalize(map1, method='range')
    map2 = normalize(map2, method='range')
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    # Compute histogram intersection
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

def cacl_emd(gtmap, salmap):
    M = ot.dist(gtmap, salmap, metric='euclidean')

    alpha = ot.unif(len(gtmap))
    beta = ot.unif(len(salmap))
    P = ot.emd(alpha, beta, M, numItermax=100000)
    pw = M * P
    # print(np.average(np.sum(pw, axis=0)))
    return np.average(np.sum(pw, axis=0))

def calc_kld(gtsAnn, resAnn):
    """
    Computer kld score. A simple implementation
    :param gtsAnn : ground-truth fixation map
    :param resAnn : predicted saliency map
    :return kld
    """

    gtsAnn = (gtsAnn - np.min(gtsAnn)) / ((np.max(gtsAnn) - np.min(gtsAnn)))
    resAnn = (resAnn - np.min(resAnn)) / ((np.max(resAnn) - np.min(resAnn)))
    eps = np.finfo(np.float32).eps
    P = resAnn / (eps + np.sum(resAnn))
    Q = gtsAnn / (eps + np.sum(gtsAnn))
    kld = np.sum(Q * np.log(eps + Q / (eps + P)))
    return kld


def cc(sals, gts, image_size=(480, 640), sigma=-1.0):
    """
    Computes CC score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param sals : dict : salmap predictions with "image name" key and ndarray as values
    :returns: average_score: float (mean CC score computed by averaging scores for all the images)
    """

    assert (len(gts) == len(sals))
    score = []
    for i in range(len(sals)):

        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)

        gtmap = Image.open(gts[i])
        gtmap = np.array(gtmap, dtype=np.float)
        if len(gtmap.shape) == 3:
            gtmap = np.mean(gtmap, axis=2)

        height_sal, width_sal = (salmap.shape[0], salmap.shape[1])
        height_fx, width_fx = (gtmap.shape[0], gtmap.shape[1])
        if image_size is None:
            salmap = scipy.ndimage.zoom(salmap,
                                        (float(height_fx) / height_sal, float(width_fx) / width_sal), order=3)
        else:
            salmap = scipy.ndimage.zoom(salmap, (float(image_size[0]) / height_sal, float(image_size[1]) / width_sal),
                                        order=3)
            gtmap = scipy.ndimage.zoom(gtmap, (float(image_size[0]) / height_fx, float(image_size[1]) / width_fx),
                                       order=3)
        if sigma > 0:
            salmap = scipy.ndimage.filters.gaussian_filter(salmap, sigma)
        salmap -= np.min(salmap)
        salmap = salmap / np.max(salmap)

        score.append(calc_cc(gtmap, salmap))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)

def nss(sals, gts, image_size=(480, 640), sigma=-1.0, fxt_field_in_mat='fixationPts', dataset='notsalicon'):
    """
    Computes NSS score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param sals : dict : saliency map predictions with "image name" key and ndarray as values
    :param image_size: [height, width]
    :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
    """
    assert(len(gts) == len(sals))

    score = []
    import matplotlib.pyplot as plt
    for i in range(len(sals)):
        # print(sals[i])
        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)

        mat = scipy.io.loadmat(gts[i])

        if dataset == 'salicon':
            fixpoint = mat['gaze']['fixations'][0][0]
            for idx in range(len(mat['gaze']['fixations'])):
                if idx > 0:
                    fixpoint = np.append(fixpoint, mat['gaze']['fixations'][idx][0], axis=0)

            all_fixpoint = np.zeros((481, 641))
            for jdx in range(len(fixpoint)):
                all_fixpoint[fixpoint[jdx][1], fixpoint[jdx][0]] = 1

            fixations = all_fixpoint
        else:
            fixations = mat[fxt_field_in_mat]
        # fixations = fixations.astype(np.bool)
        if image_size is not None:
            fixations = Image.fromarray(fixations)
            fixations = fixations.resize((image_size[1],image_size[0]), resample=Image.NEAREST)
            fixations = np.array(fixations)

        height_sal, width_sal = (salmap.shape[0], salmap.shape[1])
        if image_size is None:
            height_fx, width_fx = (fixations.shape[0], fixations.shape[1])
            salmap = scipy.ndimage.zoom(salmap, (float(height_fx)/height_sal, float(width_fx)/width_sal), order=3)
        else:
            height_fx, width_fx = (image_size[0], image_size[1])
            salmap = scipy.ndimage.zoom(salmap, (float(image_size[0])/height_sal, float(image_size[1])/width_sal), order=3)
        if sigma > 0:
            salmap = scipy.ndimage.filters.gaussian_filter(salmap, sigma)
        salmap -= np.min(salmap)
        salmap = salmap / np.max(salmap)


        score.append(calc_nss(fixations, salmap))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)


def auc(sals, gts, image_size=(480, 640), sigma=-1.0, fxt_field_in_mat='fixationPts'):
    """
    Computes AUC score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param res : dict : salmap predictions with "image name" key and ndarray as values
    :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
    """
    assert (len(gts) == len(sals))
    score = []
    for i in range(len(sals)):

        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)

        mat = scipy.io.loadmat(gts[i])
        fixations = mat[fxt_field_in_mat]
        if image_size is not None:
            fixations = Image.fromarray(fixations)
            fixations = fixations.resize((image_size[1], image_size[0]), resample=Image.NEAREST)
            fixations = np.array(fixations)

        height_sal, width_sal = (salmap.shape[0], salmap.shape[1])
        height_fx, width_fx = (fixations.shape[0], fixations.shape[1])
        salmap = scipy.ndimage.zoom(salmap, (float(height_fx) / height_sal, float(width_fx) / width_sal), order=3)
        if sigma > 0:
            salmap = scipy.ndimage.filters.gaussian_filter(salmap, sigma)
        salmap -= np.min(salmap)
        salmap = salmap / np.max(salmap)

        score.append(calc_auc(fixations, salmap))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)


def auc_salicon(sals, gts, image_size=(480, 640), sigma=-1.0, fxt_field_in_mat='fixationPts'):
    """
    Computes AUC score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param res : dict : salmap predictions with "image name" key and ndarray as values
    :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
    """
    assert (len(gts) == len(sals))
    score = []
    for i in range(len(sals)):

        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)

        mat = scipy.io.loadmat(gts[i])

        ####### kaihui salicon dataset fixation point read
        fixpoint = mat['gaze']['fixations'][0][0]
        for idx in range(len(mat['gaze']['fixations'])):
            # print('{}:{}'.format(i, mat['gaze']['fixations'][idx][0]))
            if idx > 0:
                fixpoint = np.append(fixpoint, mat['gaze']['fixations'][idx][0], axis=0)

        all_fixpoint = np.zeros((481, 641))
        for jdx in range(len(fixpoint)):
            all_fixpoint[fixpoint[jdx][1], fixpoint[jdx][0]] = 1

        fixations = all_fixpoint
        ##########

        # fixations = mat[fxt_field_in_mat]
        if image_size is not None:
            fixations = Image.fromarray(fixations)
            fixations = fixations.resize((image_size[1], image_size[0]), resample=Image.NEAREST)
            fixations = np.array(fixations)

        height_sal, width_sal = (salmap.shape[0], salmap.shape[1])
        height_fx, width_fx = (fixations.shape[0], fixations.shape[1])
        salmap = scipy.ndimage.zoom(salmap, (float(height_fx) / height_sal, float(width_fx) / width_sal), order=3)
        if sigma > 0:
            salmap = scipy.ndimage.filters.gaussian_filter(salmap, sigma)
        salmap -= np.min(salmap)
        salmap = salmap / np.max(salmap)

        score.append(calc_auc_salicon(fixations, salmap))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)

def kld(sals, gts, image_size=(480, 640)):
    """
    Computes kld score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param sals : dict : salmap predictions with "image name" key and ndarray as values
    :returns: average_score: float (mean CC score computed by averaging scores for all the images)
    """

    assert (len(gts) == len(sals))
    score = []
    for i in range(len(sals)):

        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        # print('ssssssss', salmap.shape)
        # print('ssssssss', len(salmap.shape))
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)

        gtmap = Image.open(gts[i])
        gtmap = np.array(gtmap, dtype=np.float)
        # print('ssssssss', gtmap.shape)
        # print('ssssssss', len(gtmap.shape))
        # print(image_size)
        if len(gtmap.shape) == 3:
            gtmap = np.mean(gtmap, axis=2)

        height_sal, width_sal = (salmap.shape[0], salmap.shape[1])
        height_fx, width_fx = (gtmap.shape[0], gtmap.shape[1])
        if image_size is None:
            salmap = scipy.ndimage.zoom(salmap, (float(height_fx) / height_sal, float(width_fx) / width_sal), order=3)
        else:
            salmap = scipy.ndimage.zoom(salmap, (float(image_size[0]) / height_sal, float(image_size[1]) / width_sal), order=3)
            gtmap = scipy.ndimage.zoom(gtmap, (float(image_size[0]) / height_fx, float(image_size[1]) / width_fx), order=3)

        salmap -= np.min(salmap)
        salmap = salmap / np.max(salmap)

        score.append(calc_kld(gtmap, salmap))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)

def auc_borji(sals, gts, image_size=(480, 640), fxt_field_in_mat='fixationPts', dataset='notsalicon'):
    """
    Computes AUC_borji score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param res : dict : salmap predictions with "image name" key and ndarray as values
    :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
    """
    assert (len(gts) == len(sals))
    score = []
    for i in range(len(sals)):

        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)

        mat = scipy.io.loadmat(gts[i])
        if dataset == 'salicon':
            fixpoint = mat['gaze']['fixations'][0][0]
            for idx in range(len(mat['gaze']['fixations'])):
                if idx > 0:
                    fixpoint = np.append(fixpoint, mat['gaze']['fixations'][idx][0], axis=0)

            all_fixpoint = np.zeros((481, 641))
            for jdx in range(len(fixpoint)):
                all_fixpoint[fixpoint[jdx][1], fixpoint[jdx][0]] = 1

            fixations = all_fixpoint
        else:
            fixations = mat[fxt_field_in_mat]

        if image_size is not None:
            fixations = Image.fromarray(fixations)
            fixations = fixations.resize((image_size[1], image_size[0]), resample=Image.NEAREST)
            fixations = np.array(fixations)

        height_sal, width_sal = (salmap.shape[0], salmap.shape[1])
        height_fx, width_fx = (fixations.shape[0], fixations.shape[1])
        salmap = scipy.ndimage.zoom(salmap, (float(height_fx) / height_sal, float(width_fx) / width_sal), order=3)

        salmap -= np.min(salmap)
        salmap = salmap / np.max(salmap)

        aucb, _ = calc_auc_borji(fixations, salmap)
        score.append(aucb)
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)



def auc_judd(sals, gts, image_size=(480, 640), fxt_field_in_mat='fixationPts', dataset='notsalicon'):
    """
    Computes AUC_JUDD score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param res : dict : salmap predictions with "image name" key and ndarray as values
    :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
    """
    assert (len(gts) == len(sals))
    score = []
    for i in range(len(sals)):

        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)

        mat = scipy.io.loadmat(gts[i])

        if dataset == 'salicon':
            fixpoint = mat['gaze']['fixations'][0][0]
            for idx in range(len(mat['gaze']['fixations'])):
                if idx > 0:
                    fixpoint = np.append(fixpoint, mat['gaze']['fixations'][idx][0], axis=0)

            all_fixpoint = np.zeros((481, 641))
            for jdx in range(len(fixpoint)):
                all_fixpoint[fixpoint[jdx][1], fixpoint[jdx][0]] = 1

            fixations = all_fixpoint
        else:
            fixations = mat[fxt_field_in_mat]

        if image_size is not None:
            fixations = Image.fromarray(fixations)
            fixations = fixations.resize((image_size[1], image_size[0]), resample=Image.NEAREST)
            fixations = np.array(fixations)

        height_sal, width_sal = (salmap.shape[0], salmap.shape[1])
        height_fx, width_fx = (fixations.shape[0], fixations.shape[1])
        salmap = scipy.ndimage.zoom(salmap, (float(height_fx) / height_sal, float(width_fx) / width_sal), order=3)

        salmap -= np.min(salmap)
        salmap = salmap / np.max(salmap)

        score.append(calc_auc_judd(salmap, fixations))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)

def sim(sals, gts, image_size=(480, 640)):
    """
    Computes sim score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param sals : dict : salmap predictions with "image name" key and ndarray as values
    :returns: average_score: float (mean CC score computed by averaging scores for all the images)
    """

    assert (len(gts) == len(sals))
    score = []
    for i in range(len(sals)):

        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)

        gtmap = Image.open(gts[i])
        gtmap = np.array(gtmap, dtype=np.float)
        if len(gtmap.shape) == 3:
            gtmap = np.mean(gtmap, axis=2)

        height_sal, width_sal = (salmap.shape[0], salmap.shape[1])
        height_fx, width_fx = (gtmap.shape[0], gtmap.shape[1])
        if image_size is None:
            salmap = scipy.ndimage.zoom(salmap, (float(height_fx) / height_sal, float(width_fx) / width_sal), order=3)
        else:
            salmap = scipy.ndimage.zoom(salmap, (float(image_size[0]) / height_sal, float(image_size[1]) / width_sal), order=3)
            gtmap = scipy.ndimage.zoom(gtmap, (float(image_size[0]) / height_fx, float(image_size[1]) / width_fx), order=3)
        #
        salmap -= np.min(salmap)
        salmap = salmap / np.max(salmap)

        score.append(cacl_sim(salmap, gtmap))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)

def emd(sals, gts, image_size=(480, 640)):
    """
    Computes emd score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param sals : dict : salmap predictions with "image name" key and ndarray as values
    :returns: average_score: float (mean CC score computed by averaging scores for all the images)
    """

    assert(len(gts) == len(sals))
    score = []
    for i in range(len(sals)):

        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)

        gtmap = Image.open(gts[i])
        gtmap = np.array(gtmap, dtype=np.float)
        if len(gtmap.shape) == 3:
            gtmap = np.mean(gtmap, axis=2)

        height_sal, width_sal = (salmap.shape[0], salmap.shape[1])
        height_fx, width_fx = (gtmap.shape[0], gtmap.shape[1])
        if image_size is None:
            salmap = scipy.ndimage.zoom(salmap, (float(height_fx) / height_sal, float(width_fx) / width_sal), order=3)
        else:
            salmap = scipy.ndimage.zoom(salmap, (float(image_size[0]) / height_sal, float(image_size[1]) / width_sal), order=3)
            gtmap = scipy.ndimage.zoom(gtmap, (float(image_size[0]) / height_fx, float(image_size[1]) / width_fx), order=3)

        salmap -= np.min(salmap)
        salmap = salmap / np.max(salmap)

        score.append(cacl_emd(gtmap, salmap))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)
