

import os
import pickle
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn.functional as F
from PIL import Image

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

device_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device_used}')

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


def read_data(filename: str) -> list:
    """
    :param filename: List of data points as data split.
    :return: List of data points as string.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [line.strip().split()[0] for line in lines]


def split_data(data: list, split_ratio=0.8) -> (list, list):
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


class SegmentationHead(nn.Module):
    """
    This segmentation head is attached to the model after pre-training, replacing the pre-training head.
    It consists of convolutional layers and up-sampling layers in order to get the output pixel map to match the input dimension.
    """
    def __init__(self, in_features: int, output_dim: int):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 256, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upsample4 = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.conv5 = nn.Conv2d(32, output_dim, kernel_size=3, padding=1)


    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = self.upsample1(x)
        x = F.relu(self.conv2(x))
        x = self.upsample2(x)
        x = F.relu(self.conv3(x))
        x = self.upsample3(x)
        x = F.relu(self.conv4(x))
        x = self.upsample4(x)
        x = F.relu(self.conv5(x))

        return x


class PretrainingHead(nn.Module):
    """
    This head is used for pretraining, later to be replaced by the segmentation head.
    """
    def __init__(self, in_features, output_dim):
        super(PretrainingHead, self).__init__()
        self.fc1 = nn.Linear(in_features, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class SimCLR(nn.Module):
    """
    Simple Contrastive Learning model Consists of a ResNet34 which has different heads attached for pre-training and
    segmentation.
    Paper: https://arxiv.org/abs/2002.05709
    """
    def __init__(self, feature_dim=512, out_features=512):
        super(SimCLR, self).__init__()
        self.backbone = models.resnet34(pretrained=False)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        self.head = PretrainingHead(feature_dim, out_features)

        self.flatten = True

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        if self.flatten:
            x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


# this is updated for semi-hard negative mining
class NTXentLoss(torch.nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for pre-training.
    """
    def __init__(self, temperature: float, device: torch.device):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        #self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')  # using 'none' to manually apply weights


    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        """
        Assumes z_i and z_j are normalized embeddings of shape (batch_size, feature_size). - (N,D)
        Embeddings should be from the two views of the same image (positive pairs).
        """
        batch_size = z_i.shape[0] # this is N

        # Concatenate the embeddings
        z = torch.cat((z_i, z_j), dim=0) # this is (2N,D)

        # Compute similarity
        sim = torch.mm(z, z.T) / self.temperature # this is (2N,2N)

        # Create labels
        labels = torch.arange(2 * batch_size).to(self.device)
        labels = (labels + batch_size) % (2 * batch_size) # this is (2N,)

        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        sim.masked_fill_(mask, -1e9)
        #sim.fill_diagonal_(-1e9) 

        #loss = self.criterion(sim, labels)

        # trying to implement stuff
        sim_softmax = F.softmax(sim, dim=1) # take softmax of every row - so now the values of each row (that corresponds to an anchor and its comparison with every other embedding) sums up to 1

        weights = torch.full(sim.shape, 0.3, device=self.device)  # initialize all the weights to 0.3 - has shape (2N,2N)
        
        _, indices = sim.sort(dim=1, descending=True) # this sorts every row in descending order so we have the indexes of the pairs, ranked -- the first returned thing is the sorted matrix itself which we don't need, we just need the indices 
        top_50_pct_indices = indices[:, :batch_size//2]  # get indices of the top 50% closest embeddings for each anchor - has size (2N,N)

        # Update weights for the top 50% closest (semi-hard) negatives to 0.7
        for i in range(2 * batch_size):
            weights[i, top_50_pct_indices[i]] = 0.7 # for each row, makes the relevant entries 0.7
        
        # Ensure positive pairs always contribute fully
        weights.fill_diagonal_(1.0) # similarity between each image and itself
        weights[range(batch_size, 2 * batch_size), range(batch_size)] = 1.0  # for positive pairs in the second set
        
        losses = self.criterion(sim_softmax, labels) # losses without reduction

        # Apply weights to losses
        weighted_losses = losses * weights # has shape (2N,2N), pairwise multiplication
        loss = weighted_losses.mean() # manually take the mean to return a scalar

        
        return loss


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


def pretrain(model: SimCLR, train_loader: DataLoader, optimizer: torch.optim, scheduler: StepLR, criterion: NTXentLoss, epochs=50, model_name='pretrained_model_NM') -> (SimCLR, list):
    """
    Use contrastive learning to pre-train the model.
    :param model: The model to be trained. Make sure that it has the pre-training head attached.
    :param train_loader: For the training data.
    :param optimizer: ADAM is probably the best choice.
    :param scheduler: Learning rate scheduler.
    :param criterion: NTXEnt loss function for contrastive learning.
    :param epochs: Number of epochs to train.
    :param model_name: Name of the output file without extension.
    :return: The pre-trained model and a list of losses per epoch.
    """
    model.train()

    train_loss_per_epoch = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
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

        train_loss_per_epoch.append(epoch_loss)

        scheduler.step()
        print(f'Epoch {epoch + 1} loss: {np.round(epoch_loss, 4)}')

    torch.save(model.state_dict(), f'{model_name}.pth')
    return model, train_loss_per_epoch


def finetune(model: SimCLR, train_dataloader: DataLoader, val_dataloader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim, num_epochs=50, model_name='finished_model_NM') -> (SimCLR, list, list, list, list):
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
            loss = criterion(outputs, labels)  # Adjust according to your setup
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
model, train_loss = pretrain(model, train_loader, optimizer, scheduler, criterion, epochs=50)

with open('pretraining_loss_NM.pkl', 'wb') as f:
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

model, train_loss, val_loss, train_accuracy, val_accuracy = finetune(model, oxford_train_dataloader, oxford_val_dataloader, cross_entropy_loss, optimizer, num_epochs=50)

with open('finetuning_train_loss_NM.pkl', 'wb') as f:
    pickle.dump(train_loss, f)

with open('finetuning_val_loss_NM.pkl', 'wb') as f:
    pickle.dump(val_loss, f)

with open('finetuning_train_accuracy_NM.pkl', 'wb') as f:
    pickle.dump(train_accuracy, f)

with open('finetuning_val_accuracy_NM.pkl', 'wb') as f:
    pickle.dump(val_accuracy, f)

print('Done!')

"""
# Now for the benchmark, whereby we don't pre-train and only finetune

benchmark_model = SimCLR(out_features=128).to(device)
benchmark_model.head = SegmentationHead(in_features=512, output_dim=3)

# Update the model's forward method
benchmark_model.flatten = False
optimizer = optim.Adam(benchmark_model.parameters(), lr=1e-3)

print('\n##### Begin benchmark training #####\n')

benchmark_model, train_loss, val_loss, train_accuracy, val_accuracy = finetune(benchmark_model, oxford_train_dataloader, oxford_val_dataloader, cross_entropy_loss, optimizer, num_epochs=50, model_name='benchmark')

with open('benchmark_train_loss.pkl', 'wb') as f:
    pickle.dump(train_loss, f)

with open('benchmark_val_loss.pkl', 'wb') as f:
    pickle.dump(val_loss, f)

with open('benchmark_train_accuracy.pkl', 'wb') as f:
    pickle.dump(train_accuracy, f)

with open('benchmark_val_accuracy.pkl', 'wb') as f:
    pickle.dump(val_accuracy, f)

print('Done!')
"""
