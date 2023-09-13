# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import random
import numpy as np
import torch
import torch.nn as nn
import time

random_seed = int(time.time())
random.seed(random_seed)

def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target


def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)

def renorm(img, mean, std):
    return img.mul(255.0).sub(mean).div(std)

def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks

def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask

def generate_intervals(width, num):

    assert num > 1, "The mixed strip numbers should be lager than 1"
    crop_length = int(width/num)
    crop_list = []
    for i in range(num):
        if i < num-1:
            crop_list.append((i*crop_length,(i+1)*crop_length))
        else:
            crop_list.append((i*crop_length, width))
    crop_list = random.sample(crop_list, int(num/2))
    return crop_list

def zebra_masks(target_img, num):
    shape = target_img.shape
    batch_size, height, width = shape[0], shape[2], shape[3]
    masks = []
    for i in range(batch_size):
        intervals = generate_intervals(width, num)
        mask = torch.zeros((1,1,height, width), dtype=torch.uint8).cuda()
        for intval in intervals:
            mask[:,:,:,intval[0]:intval[1]] = 1
        masks.append(mask)
    return masks



# def generate_interval(intval_list, crop_list, num):
#     # test_list.pop(random.randrange(len(test_list)))
#     intval = intval_list.pop(random.randrange(len(intval_list)))
#     crop_start = random.randint(intval[0], intval[1])
#     crop_end = random.randint(crop_start+1, intval[1])
#     crop_list.append((crop_start, crop_end))
#     if crop_start-1 > intval[0]:
#         intval_s1 = (intval[0], crop_start-1)
#         intval_list.append(intval_s1)
#     if crop_end+1 < intval[1]:
#         intval_s2 = (crop_end, intval[1])
#         intval_list.append(intval_s2)
#     if len(crop_list) < num:
#         generate_interval(intval_list, crop_list, num)
#     else:
#         return crop_list
    
# def len_intval(intval):
#     return (intval[1] - intval[0]) > 1

def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target
