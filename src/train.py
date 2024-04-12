import os
import pickle
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import *


import torch
import torch.nn.functional as F
from PIL import Image

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from config import *
from src.utils.u_datasets import read_data, split_data, ContrastiveLearningDataset, OxfordPetsDataset
from src.utils.u_models import SegmentationHead, SimCLR
from src.utils.u_train import NTXentLoss, pretrain, finetune
from src.utils.u_transformations import trans_config

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

    segmentation_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.ToTensor(),  # Convert the PIL Image to a tensor
    ])

    # Load the pre-trained model
    #model.load_state_dict(torch.load('pretrained_model.pth', map_location=torch.device('cuda')))

    # replace the pre-training head with the segmentation head
    model.head = SegmentationHead(in_features=512, output_dim=3)

    # Update the model's forward method
    model.flatten = False

    trainval_data = read_data('data/oxford/annotations/trainval.txt')
    train_data, val_data = split_data(trainval_data, split_ratio=0.8)

    batch_size = 128

    oxford_train_dataset = OxfordPetsDataset('data/oxford', train_data, transform=segmentation_transform)
    oxford_val_dataset = OxfordPetsDataset('data/oxford', val_data, transform=segmentation_transform)

    oxford_train_dataloader = DataLoader(oxford_train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    oxford_val_dataloader = DataLoader(oxford_val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    cross_entropy_loss = nn.CrossEntropyLoss()

    print('\n##### Begin fine-tuning #####\n')

    model, train_loss, val_loss, train_accuracy, val_accuracy = finetune(model, oxford_train_dataloader, oxford_val_dataloader, cross_entropy_loss, optimizer, num_epochs=50, device=device)

    with open('finetuning_train_loss.pkl', 'wb') as f:
        pickle.dump(train_loss, f)

    with open('finetuning_val_loss.pkl', 'wb') as f:
        pickle.dump(val_loss, f)

    with open('finetuning_train_accuracy.pkl', 'wb') as f:
        pickle.dump(train_accuracy, f)

    with open('finetuning_val_accuracy.pkl', 'wb') as f:
        pickle.dump(val_accuracy, f)

    # Now for the benchmark, whereby we don't pre-train and only finetune

    benchmark_model = SimCLR(out_features=128).to(device)
    benchmark_model.head = SegmentationHead(in_features=512, output_dim=3)

    # Update the model's forward method
    benchmark_model.flatten = False
    optimizer = optim.Adam(benchmark_model.parameters(), lr=1e-3)

    print('\n##### Begin benchmark training #####\n')

    benchmark_model, train_loss, val_loss, train_accuracy, val_accuracy = finetune(benchmark_model, oxford_train_dataloader, oxford_val_dataloader, cross_entropy_loss, optimizer, num_epochs=50, model_name='benchmark', device=device)

    with open('benchmark_train_loss.pkl', 'wb') as f:
        pickle.dump(train_loss, f)

    with open('benchmark_val_loss.pkl', 'wb') as f:
        pickle.dump(val_loss, f)

    with open('benchmark_train_accuracy.pkl', 'wb') as f:
        pickle.dump(train_accuracy, f)

    with open('benchmark_val_accuracy.pkl', 'wb') as f:
        pickle.dump(val_accuracy, f)

    print('Done!')

if __name__ == '__main__':
    train()
