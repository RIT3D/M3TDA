import os
import random
from loguru import logger
import numpy as np
from skimage import io, transform
import time


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
    image_a = io.imread(imgs[0])
    image_b = io.imread(imgs[1])

    # 获取图片a的高度和宽度
    height_a, width_a = image_a.shape[:2]

    # 使用时间戳作为随机数种子，确保每次运行生成不同的随机数
    random_seed = int(time.time())
    random.seed(random_seed)

    crop_start = random.randint(0, width_a - 1)
    crop_end = random.randint(crop_start + 1, width_a)


    crop_width = crop_end - crop_start

    cropped_region = image_a[:, crop_start:crop_end, :]
    cropped_region_resized = transform.resize(cropped_region, (image_b.shape[0], crop_width))


    if cropped_region_resized.shape[1] != crop_width:
        cropped_region_resized = cropped_region_resized[:, :crop_width, :]

    image_c = np.copy(image_b)
    image_c[:, crop_start:crop_end, :] = cropped_region_resized

    io.imsave('image_c_path.jpg', image_c)

def main():
    target_dataset_paths = ['./cityscapes','./gta', './Mapillary']
    selected_imgs = random_sample(target_dataset_paths)
    zebra_patch(selected_imgs)

if __name__ == "__main__":
    main()


