import os
import pickle
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
from PIL import Image

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from u_datasets import read_data, split_data, ContrastiveLearningDataset, OxfordPetsDataset
from u_models import SegmentationHead, SimCLR
from u_train import NTXentLoss, pretrain, finetune

np.set_printoptions(precision=3)

# For reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train():
    # Data Augmentation for
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

    batch_size = 2048

    train_dataset = ContrastiveLearningDataset(root_dir='./data/imagenet64', image_dim=64, transform1=transform_contrastive_1, transform2=transform_contrastive_2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    # Initialize the model and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLR(out_features=128).to(device)
    criterion = NTXentLoss(temperature=0.5, device=device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    print('\n##### Begin pre-training #####')

    # Perform pre-training
    model, train_loss = pretrain(model, train_loader, optimizer, scheduler, criterion, epochs=50, device=device)

    with open('pretraining_loss.pkl', 'wb') as f:
        pickle.dump(train_loss, f)

if __name__ == '__main__':
    train()
