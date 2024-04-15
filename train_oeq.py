import os
import pickle
import random
import numpy as np
import torch.nn as nn
from torch.nn.modules import Identity
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
from utils.u_transformations import trans_config

device_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device_used}')

#print("Current working directory:", os.getcwd())

#smaller_dir = '/tmp/pycharm_project_318/exp'
#if not os.path.exists(smaller_dir):
#    os.makedirs(smaller_dir)


def train(config_id):
    np.set_printoptions(precision=3)
    NUM_PROCS = os.cpu_count() - 2

    # For reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data Augmentation for
    transform_contrastive_1, transform_contrastive_2, id = trans_config(config_id)

    batch_size = 2048

    train_dataset = ContrastiveLearningDataset(root_dir='./data/imagenet64', image_dim=64, transform1=transform_contrastive_1, transform2=transform_contrastive_2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_PROCS)

    # Initialize the model and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLR(out_features=128).to(device)
    criterion = NTXentLoss(temperature=0.5, device=device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    print('\n##### Begin pre-training #####')

    # Perform pre-training
    model, train_loss = pretrain(model, train_loader, optimizer, scheduler, criterion, epochs=50,model_name=f"pretrained_model_oeq_{id}", device=device)

    with open(f'pretraining_loss_oeq_{id}.pkl', 'wb') as f:
        pickle.dump(train_loss, f)

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
    model.load_state_dict(torch.load(f"pretrained_model_oeq_{id}.pth", map_location=device))

    # replace the pre-training head with the segmentation head
    model.head = SegmentationHead(in_features=512, output_dim=3)

    # Update the model's forward method
    model.flatten = False

    trainval_data = read_data('data/oxford/annotations/trainval.txt')
    train_data, val_data = split_data(trainval_data, split_ratio=0.8)
    batch_size = 128

    oxford_train_dataset = OxfordPetsDataset('data/oxford', train_data, transform=segmentation_transform)
    oxford_val_dataset = OxfordPetsDataset('data/oxford', val_data, transform=segmentation_transform)

    oxford_train_dataloader = DataLoader(oxford_train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_PROCS)
    oxford_val_dataloader = DataLoader(oxford_val_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_PROCS)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    cross_entropy_loss = nn.CrossEntropyLoss()

    print('\n##### Begin fine-tuning #####\n')

    model, train_loss, val_loss, train_accuracy, val_accuracy = finetune(model, oxford_train_dataloader, oxford_val_dataloader, cross_entropy_loss, optimizer, num_epochs=50,  model_name=f'finetuned_model_oeq_{id}',device=device)

    with open(f'finetuning_train_loss_oeq_{id}.pkl', 'wb') as f:
        pickle.dump(train_loss, f)

    with open(f'finetuning_val_loss_oeq_{id}.pkl', 'wb') as f:
        pickle.dump(val_loss, f)

    with open(f'finetuning_train_accuracy_oeq_{id}.pkl', 'wb') as f:
        pickle.dump(train_accuracy, f)

    with open(f'finetuning_val_accuracy_oeq_{id}.pkl', 'wb') as f:
        pickle.dump(val_accuracy, f)


    print('Done!')

if __name__ == '__main__':
    #os.makedirs('exp', exist_ok=True)
    #train(0)
    train(1)
    train(2)
    train(3)
    train(4)
    train(5)
    train(6)
    train(7)
