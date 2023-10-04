# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from PIL import Image
from pathlib import Path

import pdb

classes_mappings_mapillary_to_cityscapes_19 = {'bird': 'other',
                                                'ground animal': 'other',
                                                'curb': 'other',
                                                'fence': 'fence',
                                                'guard rail': 'other',
                                                'barrier': 'other',
                                                'wall': 'wall',
                                                'bike lane': 'other',
                                                'crosswalk - plain': 'other',
                                                'curb cut': 'other',
                                                'parking': 'other',
                                                'pedestrian area': 'other',
                                                'rail track': 'other',
                                                'road': 'road',
                                                'service lane': 'other',
                                                'sidewalk': 'sidewalk',
                                                'bridge': 'other',
                                                'building': 'building',
                                                'tunnel': 'other',
                                                'person': 'person',
                                                'bicyclist': 'rider',
                                                'motorcyclist': 'rider',
                                                'other rider': 'rider',
                                                'lane marking - crosswalk': 'other',
                                                'lane marking - general': 'other',
                                                'mountain': 'other',
                                                'sand': 'other',
                                                'sky': 'sky',
                                                'snow': 'other',
                                                'terrain': 'terrain',
                                                'vegetation': 'vegetation',
                                                'water': 'other',
                                                'banner': 'other',
                                                'bench': 'other',
                                                'bike rack': 'other',
                                                'billboard': 'other',
                                                'catch basin': 'other',
                                                'cctv camera': 'other',
                                                'fire hydrant': 'other',
                                                'junction box': 'other',
                                                'mailbox': 'other',
                                                'manhole': 'other',
                                                'phone booth': 'other',
                                                'pothole': 'other',
                                                'street light': 'other',
                                                'pole': 'pole',
                                                'traffic sign frame': 'other',
                                                'utility pole': 'other',
                                                'traffic light': 'traffic light',
                                                'traffic sign (back)': 'traffic sign',
                                                'traffic sign (front)': 'traffic sign',
                                                'trash can': 'other',
                                                'bicycle': 'bicycle',
                                                'boat': 'other',
                                                'bus': 'bus',
                                                'car': 'car',
                                                'caravan': 'other',
                                                'motorcycle': 'motorcycle',
                                                'on rails': 'train',
                                                'other vehicle': 'other',
                                                'trailer': 'other',
                                                'truck': 'truck',
                                                'wheeled slow': 'other',
                                                'car mount': 'other',
                                                'ego vehicle': 'other',
                                                'unlabeled': 'other'}

classes_ids_19 = {'road': 0,
                'sidewalk': 1,
                'building': 2,
                'wall': 3,
                'fence': 4,
                'pole': 5,
                'traffic light': 6,
                'traffic sign': 7,
                'vegetation': 8,
                'terrain': 9,
                'sky': 10,
                'person': 11,
                'rider': 12,
                'car': 13,
                'truck': 14,
                'bus': 15,
                'train': 16,
                'motorcycle': 17,
                'bicycle': 18,
                'other': 255}

def array_from_class_mappings(dataset_classes, class_mappings, model_classes):
    """
    :param dataset_classes: list or dict. Mapping between indexes and name of classes.
                            If using a list, it's equivalent
                            to {x: i for i, x in enumerate(dataset_classes)}
    :param class_mappings: Dictionary mapping names of the dataset to
                           names of classes of the model.
    :param model_classes:  list or dict. Same as dataset_classes,
                           but for the model classes.
    :return: A numpy array representing the mapping to be done.
    """
    # Assert all classes are different.
    assert len(model_classes) == len(set(model_classes))

    # to generate the template to fill the dictionary for class_mappings
    # uncomment this code.
    """
    for x in dataset_classes:
        print((' ' * 20) + f'\'{name}\': \'\',')
    """

    # Not case sensitive to make it easier to write.
    if isinstance(dataset_classes, list):
        dataset_classes = {x: i for i, x in enumerate(dataset_classes)}
    dataset_classes = {k.lower(): v for k, v in dataset_classes.items()}
    class_mappings = {k.lower(): v.lower() for k, v in class_mappings.items()}
    if isinstance(model_classes, list):
        model_classes = {x: i for i, x in enumerate(model_classes)}
    model_classes = {k.lower(): v for k, v in model_classes.items()}

    result = np.zeros((max(dataset_classes.values()) + 1,), dtype=np.uint8)
    for dataset_class_name, i in dataset_classes.items():
        result[i] = model_classes[class_mappings[dataset_class_name]]
    return result

def convert_to_train_id(file):
    # pdb.set_trace()
    # re-assign labels to match the format of Cityscapes
    pil_lbl = Image.open(file)
    lbl = np.asarray(pil_lbl)
    labels = json.loads((
        Path('/root/feipan/M3TDA/m3tda_seg/data/mapillary/config.json')
    ).read_text())['labels']

    class_mappings = classes_mappings_mapillary_to_cityscapes_19

    model_classes = classes_ids_19

    dataset_classes = [label['readable'] for label in labels]
    vector_mappings = array_from_class_mappings(dataset_classes,
                                                class_mappings,
                                                model_classes)
    
    id_to_trainid = {}
    for i in range(66):
        id_to_trainid[i] = vector_mappings[i]

    label_copy = 255 * np.ones(lbl.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = lbl == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    new_file = file.replace('.png', '_labelTrainIds.png')
    new_file = new_file.replace('gtFine', 'gtFine2')
    # pdb.set_trace()
    assert file != new_file
    sample_class_stats['file'] = new_file
    Image.fromarray(label_copy, mode='L').save(new_file)
    # pdb.set_trace()
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GTA annotations to TrainIds')
    parser.add_argument('--mapillary_path', 
        default='/root/feipan/M3TDA/m3tda_seg/data/mapillary', help='mapillary data path')
    parser.add_argument('--gt-dir', default='gtFine/train', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=16, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)
    
    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    # pdb.set_trace()
    mapillary_path = args.mapillary_path
    out_dir = args.out_dir if args.out_dir else mapillary_path
    # pdb.set_trace()
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(mapillary_path, args.gt_dir)
    poly_files = []
    for poly in mmcv.scandir(
            gt_dir, suffix='.png',
            recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    poly_files = sorted(poly_files)

    only_postprocessing = False
    # pdb.set_trace()
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_to_train_id, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_to_train_id,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
