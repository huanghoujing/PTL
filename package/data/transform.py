import torch
import torchvision.transforms.functional as F
import random
import numpy as np
from PIL import Image
import cv2
from .random_erasing import RandomErasing

"""We expect a list `cfg.transform_list`. The types specified in this list 
will be applied sequentially. Each type name corresponds to a function name in 
this file, so you have to implement the function w.r.t. your custom type. 
The function head should be `FUNC_NAME(in_dict, cfg)`, and it should modify `in_dict`
in place.
The transform list allows us to apply optional transforms in any order, while custom
functions allow us to perform sync transformation for images and all labels.

Examples:
    transform_list = ['hflip', 'resize']
    transform_list = ['hflip', 'random_crop', 'resize']
    transform_list = ['hflip', 'resize', 'random_erase']
"""


def hflip(in_dict, cfg):
    if np.random.random() < 0.5:
        in_dict['im'] = F.hflip(in_dict['im'])
        if 'ps_label' in in_dict:
            in_dict['ps_label'] = F.hflip(in_dict['ps_label'])


def resize_3d_np_array(maps, resize_h_w, interpolation):
    """maps: np array with shape [C, H, W], dtype is not restricted"""
    return np.stack([cv2.resize(m, tuple(resize_h_w[::-1]), interpolation=interpolation) for m in maps])


def resize(in_dict, cfg):
    in_dict['im'] = Image.fromarray(cv2.resize(np.array(in_dict['im']), tuple(cfg.im.h_w[::-1]), interpolation=cv2.INTER_LINEAR))
    if 'ps_label' in in_dict:
        in_dict['ps_label'] = Image.fromarray(cv2.resize(np.array(in_dict['ps_label']), tuple(cfg.ps_label.h_w[::-1]), cv2.INTER_NEAREST), mode='L')


# If called, it should be after `to_tensor`
def random_erase(in_dict, cfg):
    in_dict['im'] = RandomErasing(probability=0.5, sh=0.2, mean=[0, 0, 0])(in_dict['im'])


def random_crop(in_dict, cfg):
    def get_params(min_keep_ratio=0.85):
        x_keep_ratio = random.uniform(min_keep_ratio, 1)
        y_keep_ratio = random.uniform(min_keep_ratio, 1)
        x1 = random.uniform(0, 1 - x_keep_ratio)
        y1 = random.uniform(0, 1 - y_keep_ratio)
        x2 = x1 + x_keep_ratio
        y2 = y1 + y_keep_ratio
        return x1, y1, x2, y2

    x1, y1, x2, y2 = get_params()
    im_w, im_h = in_dict['im'].size
    im_x1, im_y1, im_x2, im_y2 = int(im_w * x1), int(im_h * y1), int(im_w * x2), int(im_h * y2)
    in_dict['im'] = in_dict['im'].crop((im_x1, im_y1, im_x2, im_y2))
    if 'ps_label' in in_dict:
        ps_w, ps_h = in_dict['ps_label'].size
        ps_x1, ps_y1, ps_x2, ps_y2 = int(ps_w * x1), int(ps_h * y1), int(ps_w * x2), int(ps_h * y2)
        in_dict['ps_label'] = in_dict['ps_label'].crop((ps_x1, ps_y1, ps_x2, ps_y2))
    return in_dict

##########################
# Enhancement Augmentation

from PIL import Image, ImageFilter

def blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

all_augs = [
    (blur, [1,2]),
    (F.adjust_brightness, [0.6, 1.4]),
    (F.adjust_contrast, [0.6, 1.4]),
    (F.adjust_saturation, [0.6, 1.4]),
    (F.adjust_gamma, [0.6, 1.4]),
]

def aug_im(im, func, param, prob):
    if np.random.rand() < prob:
        im = func(im, param)
    return im

def comb_aug_im(im, all_augs):
    if np.random.rand() < 0.5:
        return im
    random.shuffle(all_augs)
    for func, ran in all_augs:
        param = np.random.choice(ran) if func is blur else np.random.uniform(ran[0], ran[1])
        im = aug_im(im, func, param, 0.7)
    return im

def enhance_aug(in_dict, cfg):
    in_dict['im'] = comb_aug_im(in_dict['im'], all_augs)

##########################


def to_tensor(in_dict, cfg):
    in_dict['im'] = F.to_tensor(in_dict['im'])
    in_dict['im'] = F.normalize(in_dict['im'], cfg.im.mean, cfg.im.std)
    if 'ps_label' in in_dict:
        in_dict['ps_label'] = torch.from_numpy(np.array(in_dict['ps_label'])).long()


def transform(in_dict, cfg):
    for t in cfg.transform_list:
        if t != 'random_erase':
            eval('{}(in_dict, cfg)'.format(t))
    to_tensor(in_dict, cfg)
    if 'random_erase' in cfg.transform_list:
        random_erase(in_dict, cfg)
    return in_dict
