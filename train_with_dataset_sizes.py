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

from utils.u_datasets import read_data, split_data, ContrastiveLearningDataset, OxfordPetsDataset
from utils.u_models import SegmentationHead, SimCLR
from utils.u_train import NTXentLoss, pretrain, finetune


def main(train_size):
    """
    Trains a model with and without pretraining.

    :param train_size: The size of the fine-tuning set.
    """
    np.set_printoptions(precision=3)

    # For reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    NUM_PROCS = os.cpu_count() - 2

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    segmentation_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.ToTensor(),  # Convert the PIL Image to a tensor
    ])

    # Load the pre-trained model
    model = SimCLR(out_features=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.load_state_dict(torch.load('pretrained_model.pth', map_location=device))

    # replace the pre-training head with the segmentation head
    model.head = SegmentationHead(in_features=512, output_dim=3)

    # Update the model's forward method
    model.flatten = False

    trainval_data = read_data('data/oxford/annotations/trainval.txt')
    train_data, val_data = split_data(trainval_data, split_ratio=0.8)
    train_data, _ = split_data(train_data, split_ratio=train_size)
    batch_size = 128

    oxford_train_dataset = OxfordPetsDataset('data/oxford', train_data, transform=segmentation_transform)
    oxford_val_dataset = OxfordPetsDataset('data/oxford', val_data, transform=segmentation_transform)

    oxford_train_dataloader = DataLoader(oxford_train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_PROCS)
    oxford_val_dataloader = DataLoader(oxford_val_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_PROCS)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    cross_entropy_loss = nn.CrossEntropyLoss()

    print('\n##### Begin fine-tuning #####\n')

    model, train_loss, val_loss, train_accuracy, val_accuracy = finetune(model, oxford_train_dataloader, oxford_val_dataloader, cross_entropy_loss, optimizer, num_epochs=50, model_name=f"pretrained_{train_size}",device=device)

    with open(f'finetuning_train_loss_{train_size}.pkl', 'wb') as f:
        pickle.dump(train_loss, f)
    with open(f'finetuning_val_loss_{train_size}.pkl', 'wb') as f:
        pickle.dump(val_loss, f)

    with open(f'finetuning_train_accuracy_{train_size}.pkl', 'wb') as f:
        pickle.dump(train_accuracy, f)

    with open(f'finetuning_val_accuracy_{train_size}.pkl', 'wb') as f:
        pickle.dump(val_accuracy, f)

    # Now for the benchmark, whereby we don't pre-train and only finetune

    benchmark_model = SimCLR(out_features=128).to(device)
    benchmark_model.head = SegmentationHead(in_features=512, output_dim=3)

    # Update the model's forward method
    benchmark_model.flatten = False
    optimizer = optim.Adam(benchmark_model.parameters(), lr=1e-3)

    print('\n##### Begin benchmark training #####\n')

    benchmark_model, train_loss, val_loss, train_accuracy, val_accuracy = finetune(benchmark_model, oxford_train_dataloader, oxford_val_dataloader, cross_entropy_loss, optimizer, num_epochs=50, model_name=f'benchmark_{train_size}', device=device)

    with open(f'benchmark_train_loss_{train_size}.pkl', 'wb') as f:
        pickle.dump(train_loss, f)

    with open(f'benchmark_val_loss_{train_size}.pkl', 'wb') as f:
        pickle.dump(val_loss, f)

    with open(f'benchmark_train_accuracy_{train_size}.pkl', 'wb') as f:
        pickle.dump(train_accuracy, f)

    with open(f'benchmark_val_accuracy_{train_size}.pkl', 'wb') as f:
        pickle.dump(val_accuracy, f)

    print('Done!')

if __name__ == '__main__':
    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        main(i)
