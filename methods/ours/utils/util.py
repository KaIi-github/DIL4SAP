import os
import sys

import numpy as np


def step_lr(epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
    """Handles step decay of learning rate."""
    factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
    new_lr = base_lr * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print('Set lr to ', new_lr)
    return optimizer

def detect_images_type(folder):
    images = []
    if not os.path.isdir(folder):
        raise Exception('{} does not exist'.format(folder))
        sys.exit(-1)
    for root, _, fnames in sorted(os.walk(folder)):
        for fname in sorted(fnames):
            tmp_type = fname[fname.find('.')+1:]
            if tmp_type in ['jpg', 'jpeg', 'png', 'ppm', 'bmp']:
                return tmp_type
        break
    return ''

def load_allimages_list(dir):
    images = []
    if not os.path.isdir(dir):
        raise Exception('{} does not exist'.format(dir))
        sys.exit(-1)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if (fname.endswith('.jpg') or 
            	fname.endswith('.jpeg') or
            	fname.endswith('.png') or
            	fname.endswith('.ppm') or
            	fname.endswith('.bmp')):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images

def load_allimages_list_norec(dir):
    images = []
    if not os.path.isdir(dir):
        raise Exception('{} does not exist'.format(dir))
        sys.exit(-1)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if (fname.endswith('.jpg') or 
                fname.endswith('.jpeg') or
                fname.endswith('.png') or
                fname.endswith('.ppm') or
                fname.endswith('.bmp')):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
        break
    return images

def load_allimages_wopath(dir):
    images = []
    if not os.path.isdir(dir):
        raise Exception('{} does not exist'.format(dir))
        sys.exit(-1)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if (fname.endswith('.jpg') or 
                fname.endswith('.jpeg') or
                fname.endswith('.png') or
                fname.endswith('.ppm') or
                fname.endswith('.bmp')):
                images.append(fname)
        break
    return images

def load_allmats(dir):
    images = []
    if not os.path.isdir(dir):
        raise Exception('failed to load mats in folder {}'.format(dir))
        sys.exit(-1)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if (fname.endswith('.mat')):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images

def load_allimages(dir):
    images = []
    if not os.path.isdir(dir):
        raise Exception('failed to load images in folder {}'.format(dir))
        sys.exit(-1)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if (fname.endswith('.jpg') or 
                fname.endswith('.jpeg') or
                fname.endswith('.png') or
                fname.endswith('.ppm') or
                fname.endswith('.bmp')):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)
        break
    return images

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

