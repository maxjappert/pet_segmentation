import os
import pickle
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import torch
from PIL import Image

def read_data(filename: str) -> list:
    """
    :param filename: List of data points as data split.
    :return: List of data points as string.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [line.strip().split()[0] for line in lines]


def split_data(data: list, split_ratio=0.8) -> tuple[list, list]:
    """
    Splits data in two.
    :param data: List of string data points.
    :param split_ratio: Ratio of split.
    :param seed: Seed for random split.
    :return: Two lists, each a random split of the original data.
    """
    random.shuffle(data)  # Randomly shuffle data
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]

def unpickle(file: str) -> dict:
    """
    For the data sets.
    :param file: Pickle file.
    :return: Dictionary with unpickled data.
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

class ContrastiveLearningDataset(Dataset):
    """
    ImageNet dataset for the pre-training. Returns two versions of each image, each transformed differently.
    """
    def __init__(self, root_dir: str, image_dim=64, transform1=None, transform2=None):
        """
        :param root_dir: Data directory.
        :param image_dim: Height/width of the images. We're using ImageNet, which always has square images.
        :param transform1: Transform for the image 1.
        :param transform2: Transform for the image 2.
        """
        self.root_dir = root_dir
        self.transform1 = transform1
        self.transform2 = transform2
        self.images = []

        for file in os.listdir(root_dir):
            if file.__contains__('batch'):
                dictionary = unpickle(os.path.join(root_dir, file))
                for image in dictionary['data']:
                    reshaped = np.array(image).reshape((3, image_dim, image_dim))
                    self.images.append(Image.fromarray(reshaped.transpose((1, 2, 0))).convert('RGB'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        if self.transform1 and self.transform2:
            image1 = self.transform1(image)
            image2 = self.transform2(image)
        else:
            image1, image2 = image, image

        return image1, image2


class OxfordPetsDataset(Dataset):
    """
    Dataset for the Oxford Pets data.
    """
    def __init__(self, root_dir: str, data: list, transform=None):
        """
        :param root_dir: Data dir
        :param data: List of data names. Can be extracted from the data/oxford/annotations/*.txt files.
        :param transform: Transform for the image.
        """
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_dir = os.path.join(root_dir, 'annotations/trimaps')
        self.transform = transform
        self.images = [datum+'.jpg' for datum in data]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        annotation_path = os.path.join(self.annotation_dir, img_name.replace(".jpg", ".png"))
        image = Image.open(img_path).convert("RGB")
        annotation = Image.open(annotation_path).convert("L")

        if self.transform:
            image = self.transform(image)
            normaliser = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normaliser(image)
            annotation = (self.transform(annotation)*255)-1
            annotation = torch.squeeze(annotation, 0).long()

        return image, annotation