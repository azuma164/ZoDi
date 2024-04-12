# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:51:21 2019 by Attila Lengyel - attila@lengyel.nl
"""

import os
import numpy as np

from PIL import Image

from torchvision.datasets import Cityscapes
import torch
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, maxImgNum=1e5, name=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) and ((name in fname) if name is not None else 1):
                path = os.path.join(root, fname)
                images.append(path)
                # print(path)
    # random.shuffle(images)
    print(len(images))
    return images[:min(maxImgNum, len(images))]

class CityTwoDomains(Cityscapes):
    
    voidClass = 19

    # Convert ids to train_ids
    id2trainid = np.array([label.train_id for label in Cityscapes.classes if label.train_id >= 0], dtype='uint8')
    id2trainid[np.where(id2trainid==255)] = voidClass

    # Convert train_ids to ids
    trainid2id = np.arange(len(id2trainid))[np.argsort(id2trainid)]
    
    # Convert train_ids to colors
    mask_colors = [list(label.color) for label in Cityscapes.classes if label.train_id >= 0 and label.train_id <= 19]
    mask_colors.append([0,0,0])
    mask_colors = np.array(mask_colors)
    
    # List of valid class ids
    validClasses = np.unique([label.train_id for label in Cityscapes.classes if label.id >= 0])
    validClasses[np.where(validClasses==255)] = voidClass
    validClasses = list(validClasses)
    
    # Create list of class names
    classLabels = [label.name for label in Cityscapes.classes if not (label.ignore_in_eval or label.id < 0)]
    classLabels.append('void')

    def __init__(self, root, root_new_domain=None, split='train', target_type='semantic', transforms=None):
        super().__init__(root, split=split, target_type=target_type, transforms=transforms)
        self.root_new_domain = root_new_domain
        self.images_new_domain = []

        for city in os.listdir(self.root_new_domain):
            img_dir = os.path.join(self.root_new_domain, city)
            for file_name in os.listdir(img_dir):
                self.images_new_domain.append(os.path.join(img_dir, file_name))

        assert (len(self.images) == len(self.images_new_domain))


    def __getitem__(self, index):
        filepath = self.images[index]
        image = Image.open(filepath).convert('RGB')
        filepath_new_domain = self.images_new_domain[index]
        image_new_domain = Image.open(filepath_new_domain).convert('RGB')

        # print(filepath, filepathZurich)

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, image_new_domain, target = self.transforms(image, image_new_domain, target)
            
        target = self.id2trainid[target] # Convert class ids to train_ids and then to tensor: SLOW

        return image, image_new_domain, target
    