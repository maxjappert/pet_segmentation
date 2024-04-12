import os
import pickle
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import torch
import torch.nn.functional as F
from PIL import Image

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from u_datasets import read_data, split_data, ContrastiveLearningDataset, OxfordPetsDataset
from u_models import SegmentationHead, SimCLR
from u_train import NTXentLoss, pretrain, finetune
from u_transformations import trans_config

def train(config_id):
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

    transform_contrastive_1, transform_contrastive_2, id = trans_config(config_id)
    # Data Augmentation for

    batch_size = 2048

    dataset = ContrastiveLearningDataset(root_dir='./data/imagenet64', image_dim=64, transform1=transform_contrastive_1, transform2=transform_contrastive_2)
    dataset_size = len(dataset)  # Total number of examples in the dataset
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, num_workers=12)  # Usually, shuffle is False for validation/test loaders


    # Initialize the model and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLR(out_features=128).to(device)
    criterion = NTXentLoss(temperature=0.5, device=device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    print('\n##### Begin pre-training #####')

    # Perform pre-training
    model, train_loss, val_loss = pretrain(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs=50, device=device)

    with open('pretraining_train_loss.pkl', 'wb') as f:
        pickle.dump(train_loss, f)

    with open('pretraining_val_loss.pkl', 'wb') as f:
        pickle.dump(val_loss, f)


if __name__ == '__main__':
    train(0)
    train(1)
    train(2)
    train(3)
    train(4)
    train(5)
    train(6)
    train(7)
