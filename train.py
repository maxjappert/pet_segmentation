import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


# Define the SimCLR model
class SimCLR(nn.Module):
    def __init__(self, out_features=1024):
        super(SimCLR, self).__init__()
        self.backbone = models.resnet18(pretrained=False)  # Use a ResNet-50 model        dim_mlp = self.backbone.fc.in_features  # Get features before the fully connected layer
        dim_mlp = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(  # Replace the fc layer for projection head
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_features),
        )

    def forward(self, x):
        return self.backbone(x)


# Contrastive loss function
class NTXentLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        # Create a mask to exclude positive samples from the numerator and avoid computing similarity with itself
        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity_function = self._cosine_similarity
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        # We use a mask to exclude the positive examples from the denominator
        # This mask will filter out the positive examples and leave only the negatives
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool).fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask.to(self.device)

    def _cosine_similarity(self, z_i, z_j):
        # Normalize the input features
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # Compute the cosine similarity
        return torch.mm(z_i, z_j.t())

    def forward(self, z_i, z_j):
        # Concatenate the features from both views
        features = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = self.similarity_function(features, features)

        # Mask out the positive samples
        similarity_matrix = similarity_matrix.masked_select(self.mask).view(2 * self.batch_size, -1)

        # Scale the similarity with the temperature
        similarity_matrix /= self.temperature

        # The labels for the CrossEntropyLoss are the indices of the positive samples in the similarity matrix
        labels = torch.arange(self.batch_size).to(self.device)
        labels = torch.cat([labels, labels], dim=0)

        loss = self.criterion(similarity_matrix, labels)
        return loss / (2 * self.batch_size)  # Normalize the loss by the batch size


def train(model, train_loader, optimizer, scheduler, device, epochs=10):
    model.train()  # Set the model to training mode

    # Loop over the dataset multiple times
    for epoch in range(epochs):
        epoch_loss = 0.0

        for i, (images, _) in enumerate(train_loader, 0):
            # Get a batch of images and their labels (labels are not used here)
            images = torch.cat(images, dim=0)  # Concatenate images to form a batch of pairs
            images = images.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            z_i, z_j = model(images[:len(images)//2]), model(images[len(images)//2:])
            loss = criterion(z_i, z_j)  # Compute the loss

            # Backward pass + optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            epoch_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {epoch_loss / 100}')
                epoch_loss = 0.0

        # Step the scheduler
        scheduler.step()
        print(f'Epoch {epoch + 1} finished')


# TODO: Add other augmentation
# Data Augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = 128

# TODO: Replace with ImageNet and filter out animal images
# Load the dataset (example with CIFAR10 for simplicity)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLR(out_features=128).to(device)
criterion = NTXentLoss(temperature=0.5, device=device, batch_size=batch_size)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Assuming the model, optimizer, scheduler, and NTXentLoss ('criterion') are defined
# and 'device' is set to 'cuda' if available
train(model, train_loader, optimizer, scheduler, device, epochs=10)
