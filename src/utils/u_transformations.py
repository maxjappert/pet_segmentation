import os
import pickle
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from config import *


def trans_config(num):
    """
    Returns the data augmentation transformations for the contrastive learning task.
    0: Original paper ( , )
    1: (+Elastic Transform, )
    2: (+Random Posterize, )
    3: (+Random Solarize, )
    4: (+Elastic Transform+Random Posterize, )
    5: (+Elastic Transform+Random Solarize, )
    6: (+Random Posterize+Random Solarize, )
    7: (+Elastic Transform+Elastic Transform+Random Posterize+Random Solarize, )

    Args:
        num (int): The config number
    """
    if num == 0:
        transform_contrastive_1 = transforms.Compose([
            transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_contrastive_2 = transforms.Compose([
            transforms.RandomResizedCrop(size=64),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        id = ""

    elif num == 1:
        transform_contrastive_1 = transforms.Compose([
            transforms.ElasticTransform(),
            transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_contrastive_2 = transforms.Compose([
            transforms.RandomResizedCrop(size=64),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        id = "ET"

    elif num == 2:
        transform_contrastive_1 = transforms.Compose([
            transforms.RandomPosterize(bits=4), # Is this right?
            transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_contrastive_2 = transforms.Compose([
            transforms.RandomResizedCrop(size=64),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        id = "RP"

    elif num == 3:
        transform_contrastive_1 = transforms.Compose([
            transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),
            transforms.RandomSolarize(threshold=192), # Is this right?
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_contrastive_2 = transforms.Compose([
            transforms.RandomResizedCrop(size=64),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        id = "RS"

    elif num == 4:
        transform_contrastive_1 = transforms.Compose([
            transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),
            transforms.ElasticTransform(),
            transforms.RandomPosterize(bits=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_contrastive_2 = transforms.Compose([
            transforms.RandomResizedCrop(size=64),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        id = "ET_RP"

    elif num == 5:
        transform_contrastive_1 = transforms.Compose([
            transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),
            transforms.ElasticTransform(),
            transforms.RandomSolarize(threshold=192),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_contrastive_2 = transforms.Compose([
            transforms.RandomResizedCrop(size=64),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        id = "ET_RS"

    elif num == 6:
        transform_contrastive_1 = transforms.Compose([
            transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),
            transforms.RandomSolarize(threshold=192),
            transforms.RandomPosterize(bits=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_contrastive_2 = transforms.Compose([
            transforms.RandomResizedCrop(size=64),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        id = "RP_RS"

    elif num == 7:
        transform_contrastive_1 = transforms.Compose([
            transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),
            transforms.ElasticTransform(),
            transforms.RandomPosterize(bits=4),
            transforms.RandomSolarize(threshold=192),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_contrastive_2 = transforms.Compose([
            transforms.RandomResizedCrop(size=64),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        id = "ET+RP+RS_"

    else:
        print("Invalid augmentations ID")
        transform_contrastive_1, transform_contrastive_2 = None, None
        id = "no_transform"

    return transform_contrastive_1, transform_contrastive_2, id
