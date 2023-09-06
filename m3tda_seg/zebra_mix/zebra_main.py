import os
import random
from loguru import logger
import numpy as np
from skimage import io, transform


def random_sample(target_dataset_paths):
    # Randomly sample two images from different target datasets 
    img_list = []
    for target_dataset in target_dataset_paths:
        imgs = os.listdir(target_dataset)
        imgs = [os.path.join(target_dataset, img) for img in imgs]
        img_list.extend(imgs)
    while True:
        selected_imgs = random.sample(img_list, 2)
        if os.path.split(selected_imgs[0])[0] != os.path.split(selected_imgs[1])[0]:
            break

    logger.info("selected images: {}",selected_imgs)
    return selected_imgs

def zebra_patch(imgs):
    # paste zebra(vertical) patch on image  
    image_back = io.imread(imgs[0])
    image_patch = io.imread(imgs[1])

    # resize first
    image_patch_resized = transform.resize(image_patch, (image_back.shape[0], image_back.shape[1]))*255
    width_patch = image_patch.shape[1]

    # crop image as patch；leave margin on both sides
    crop_start = random.randint(int(width_patch* 0.2), int(width_patch* 0.8))
    crop_end = random.randint(int(crop_start + 5), int(width_patch*0.95))
    cropped_region = image_patch_resized[:, crop_start:crop_end, :]
  
    # paste patch on image
    image_zebra = np.copy(image_back)
    image_zebra[:, crop_start:crop_end, :] = cropped_region
    io.imsave('image_zebra.jpg', image_zebra)

def main():
    target_dataset_paths = ['./cityscapes','./gta', './Mapillary']
    selected_imgs = random_sample(target_dataset_paths)
    zebra_patch(selected_imgs)

if __name__ == "__main__":
    main()


