import random
import pdb
import torch
from PIL import Image
import numpy as np
import time


def generate_intervals(width, num):
    random_seed = int(time.time())
    random.seed(random_seed)
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
        mask = torch.zeros((1,1,height, width)).cuda()
        for intval in intervals:
            mask[:,:,:,intval[0]:intval[1]] = 1
        masks.append(mask)
    return masks, intervals


if __name__ == '__main__':
    img1 = Image.open('/root/feipan/M3TDA/m3tda_seg/zebra_mix/cityscapes/berlin_000001_000019_leftImg8bit.png')
    img1=img1.resize((1024,512))
    img2 = Image.open('/root/feipan/M3TDA/m3tda_seg/zebra_mix/gta/00001.png')
    img2=img2.resize((1024,512))
    img1 = torch.tensor(np.asarray(img1).transpose(2,0,1)).cuda()
    img1 = img1[None,:]
    img2 = torch.tensor(np.asarray(img2).transpose(2,0,1)).cuda()
    img2 = img2[None,:]
    masks, intervals = zebra_masks(img1, num=4)
    mask = masks[0]
    mask = torch.cat((mask,mask,mask), dim=1)
    new_img = img1 * mask + img2 *(1-mask)
    new_img = new_img.cpu().numpy().astype(np.uint8)[0].transpose(1,2,0)
    new_img = Image.fromarray(new_img)
    print(intervals)
    new_img.save('output.png')
