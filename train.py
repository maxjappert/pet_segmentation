import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn.functional as F
from PIL import Image

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import requests, zipfile, io

# Example of adding a segmentation head to the SimCLR model (highly simplified)
class SegmentationHead(nn.Module):
    def __init__(self, in_features, output_dim):
        super(SegmentationHead, self).__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(in_features, 512, kernel_size=3, padding=1)

        # Upsampling layer
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        # Further processing and final upsample to match input size
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.upsample3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.upsample5 = nn.ConvTranspose2d(16, output_dim, kernel_size=2, stride=2)

# TODO: annotations seem not correct, must be integers

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.upsample1(x)
        x = F.relu(self.conv2(x))
        x = self.upsample2(x)  # Output size [N, num_classes, H, W]
        x = self.upsample3(x)  # Output size [N, num_classes, H, W]
        x = self.upsample4(x)  # Output size [N, num_classes, H, W]
        x = self.upsample5(x)  # Output size [N, num_classes, H, W]
        return x


# Example of adding a segmentation head to the SimCLR model (highly simplified)
class PretrainingHead(nn.Module):
    def __init__(self, in_features, output_dim):
        super(PretrainingHead, self).__init__()
        self.fc = nn.Linear(in_features, 512)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.fc(x)  # Output size [num_classes, H, W]
        return x

# Define the SimCLR model
class SimCLR(nn.Module):
    def __init__(self, feature_dim=512, out_features=512):
        super(SimCLR, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Output channels from the backbone's last conv layer (512 for ResNet18/34; 2048 for ResNet50/101/152)
        in_features = 512

        self.head = PretrainingHead(feature_dim, out_features)

        self.flatten = True

    def forward(self, x):
        x = self.backbone(x)  # Pass input through the backbone
        if self.flatten:
            x = x.view(x.size(0), -1)  # Flatten the output for the pretraining head
        x = self.head(x)  # Pass through the pretraining head
        return x


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature, device):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        Assumes z_i and z_j are normalized embeddings of shape (batch_size, feature_size).
        Embeddings should be from the two views of the same image (positive pairs).
        """
        batch_size = z_i.shape[0]

        # Concatenate the embeddings
        z = torch.cat((z_i, z_j), dim=0)

        # Compute similarity
        sim = torch.mm(z, z.T) / self.temperature

        # Create labels
        labels = torch.arange(2 * batch_size).to(self.device)
        labels = (labels + batch_size) % (2 * batch_size)

        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        sim.masked_fill_(mask, -1e9)

        loss = self.criterion(sim, labels)
        return loss


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


class ContrastiveLearningDataset(Dataset):
    def __init__(self, root_dir, transform1=None, transform2=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform1 = transform1
        self.transform2 = transform2
        self.images = []

        for file in os.listdir(root_dir):
            if file.__contains__('batch'):
                dictionary = unpickle(os.path.join(root_dir, file))
                #self.images += list(dictionary['data'])
                #batch = dictionary['data']
                #batch.reshape((batch.shape[0], 3, 8, 8))
                for image in dictionary['data']:
                    reshaped = np.array(image).reshape((3, 8, 8))
                    self.images.append(Image.fromarray(reshaped.transpose((1, 2, 0))).convert('RGB'))
                # TODO: Remove break
                break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform1 and self.transform2:
            image1 = self.transform1(image)
            image2 = self.transform2(image)
        else:
            image1, image2 = image, image

        return image1, image2

class OxfordPetsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_dir = os.path.join(root_dir, 'annotations/trimaps')
        self.transform = transform
        self.images = os.listdir(self.image_dir)
        self.images = [image for image in self.images if '.jpg' in image]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        annotation_path = os.path.join(self.annotation_dir, img_name.replace(".jpg", ".png"))
        image = Image.open(img_path).convert("RGB")
        annotation = Image.open(annotation_path)

        if self.transform is not None:
            image = self.transform(image)
            normaliser = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normaliser(image)
            annotation = self.transform(annotation)
            #print(annotation.max())
            #print(annotation.min())

        return image, annotation


def pretrain(model, train_loader, optimizer, scheduler, criterion, epochs=10, model_name='pretrained_model'):
    model.train()  # Set the model to training mode

    # Loop over the dataset multiple times
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for i, (image1, image2) in enumerate(train_loader):
            # Get a batch of images and their labels (labels are not used here)
            image1 = image1.to(device)
            image2 = image2.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            z_i, z_j = model(image1), model(image2)
            loss = criterion(z_i, z_j)  # Compute the loss

            # Backward pass + optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            epoch_loss += loss.item()
            num_batches += 1
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {np.round(epoch_loss / num_batches, 3)}')
                epoch_loss = 0.0
                num_batches = 0
        # Step the scheduler
        scheduler.step()
        print(f'Epoch {epoch + 1} finished')

    torch.save(model.state_dict(), f'{model_name}.pth')
    return model


def finetune(model, dataloader, criterion, optimizer, num_epochs=10, model_name='finished_model'):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in dataloader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # Adjust according to your setup
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}')

    print('Training complete')

    torch.save(model.state_dict(), f'{model_name}.pth')
    return model


# TODO: Add other augmentation
# Data Augmentation
# Define your two sets of transformations
transform1 = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform2 = transforms.Compose([
    transforms.RandomResizedCrop(size=256),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = 128

# TODO: Replace with ImageNet and filter out animal images
# Load the dataset (example with CIFAR10 for simplicity)
#datasets.CIFAR10(root='./data', train=True, download=True)
train_dataset = ContrastiveLearningDataset(root_dir='./data/imagenet8', transform1=transform1, transform2=transform2)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

# Initialize the model and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLR(out_features=128).to(device)
criterion = NTXentLoss(temperature=0.5, device=device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Assuming the model, optimizer, scheduler, and NTXentLoss ('criterion') are defined
# and 'device' is set to 'cuda' if available
#model = pretrain(model, train_loader, optimizer, scheduler, criterion, epochs=1)

# Example usage
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert the PIL Image to a tensor
])

model.load_state_dict(torch.load('pretrained_model.pth', map_location=torch.device('cuda')))

# Assuming model is your instance of SimCLR
model.backbone = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-2])  # Reinitialize or ensure backbone is correct
model.head = SegmentationHead(in_features=512, output_dim=1)  # For binary segmentation, adjust output_dim as needed

# Update the model's forward method
model.flatten = False

oxford_dataset = OxfordPetsDataset(root_dir='data/oxford', transform=transform)
oxford_dataloader = DataLoader(oxford_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

cross_entropy_loss = nn.CrossEntropyLoss()

model = finetune(model, oxford_dataloader, cross_entropy_loss, optimizer, num_epochs=10)