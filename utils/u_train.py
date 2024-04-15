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

from utils.u_models import SimCLR

class NTXentLoss(torch.nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for pre-training.
    """
    def __init__(self, temperature: float, device: torch.device):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        """
        Assumes z_i and z_j are normalized embeddings of shape (batch_size, feature_size).
        Embeddings should be from the two views of the same image (positive pairs).
        """
        batch_size = z_i.shape[0]

        # Concatenate the embeddings
        z = torch.cat((z_i, z_j), dim=0).to(self.device)

        # Compute similarity
        sim = (torch.mm(z, z.T) / self.temperature).to(self.device)

        # Create labels
        labels = torch.arange(2 * batch_size)
        labels = ((labels + batch_size) % (2 * batch_size)).to(self.device)

        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        sim.masked_fill_(mask, -1e9)

        loss = self.criterion(sim, labels)
        return loss

def pretrain(model: SimCLR, train_loader: DataLoader, optimizer: torch.optim, scheduler: StepLR, criterion: NTXentLoss, epochs=50, model_name='pretrained_model', device = 'cpu') -> tuple[SimCLR, list, list]:
    """
    Use contrastive learning to pre-train the model.
    :param model: The model to be trained. Make sure that it has the pre-training head attached.
    :param train_loader: For the training data.
    :param optimizer: ADAM is probably the best choice.
    :param scheduler: Learning rate scheduler.
    :param criterion: NTXEnt loss function for contrastive learning.
    :param epochs: Number of epochs to train.
    :param model_name: Name of the output file without extension.
    :param device: Device to run computation on.
    :return: The pre-trained model and a list of losses per epoch.
    """

    model = model.to(device)

    train_loss_per_epoch = []
    val_loss_per_epoch = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for i, (image1, image2) in enumerate(train_loader):

            image1 = image1.to(device)
            image2 = image2.to(device)

            optimizer.zero_grad()

            z_i, z_j = model(image1), model(image2)
            loss = criterion(z_i, z_j)

            loss.backward()
            optimizer.step()

            # Print statistics
            epoch_loss += loss.item()

        train_loss_per_epoch.append(epoch_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image1, image2 in val_loader:
                image1, image2 = image1.to(device), image2.to(device)
                z_i, z_j = model(image1), model(image2)
                loss = criterion(z_i, z_j)
                val_loss += loss.item()

        val_loss_per_epoch.append(val_loss / len(val_loader))

        scheduler.step()
        print(f'Epoch {epoch + 1}:')
        print(f'Train loss: {np.round(epoch_loss / len(train_loader), 4)}')
        print(f'Val loss: {np.round(val_loss / len(val_loader), 4)}')

    torch.save(model.state_dict(), f'{model_name}.pth')
    return model, train_loss_per_epoch, val_loss_per_epoch


def finetune(model: SimCLR, train_dataloader: DataLoader, val_dataloader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim, num_epochs=50, model_name='finished_model', device = 'cpu') -> tuple[SimCLR, list, list, list, list]:
    """
    To finetune the model after pre-training.
    :param model: The model to be fine-tuned. Make sure to replace the pre-training head with the segmentation head
    before fine-tuning.
    :param train_dataloader: For the training data.
    :param val_dataloader: For the validation data.
    :param criterion: Cross entropy loss function for segmentation.
    :param optimizer: Optimizer.
    :param num_epochs: Number of epochs.
    :param model_name: For output file, without extension.
    :param device: Device to run computation on.
    :return: Fine-tuned model with lists of losses and accuracies per epoch for both training and validation data.
    """
    model = model.to(device)
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    train_classification_accuracy_per_epoch = []
    val_classification_accuracy_per_epoch = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_dataloader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)  # Adjust according to your setup
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)*labels.size(1)*labels.size(2)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)*labels.size(1)*labels.size(2)
                val_correct += (predicted == labels).sum().item()

        train_epoch_loss = train_loss / len(train_dataloader)
        val_epoch_loss = val_loss / len(val_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train loss: {np.round(train_epoch_loss, 3)}')
        print(f'Val loss: {np.round(val_epoch_loss, 3)}')
        print(f'Train classification accuracy: {np.round((train_correct/train_total)*100, 3)}%')
        print(f'Validation classification accuracy: {np.round((val_correct/val_total)*100, 3)}%')
        train_loss_per_epoch.append(train_epoch_loss)
        val_loss_per_epoch.append(val_epoch_loss)
        train_classification_accuracy_per_epoch.append(train_correct/train_total)
        val_classification_accuracy_per_epoch.append(val_correct/val_total)

    print('Training complete')

    torch.save(model.state_dict(), f'{model_name}.pth')
    return model, train_loss_per_epoch, val_loss_per_epoch, train_classification_accuracy_per_epoch, val_classification_accuracy_per_epoch
