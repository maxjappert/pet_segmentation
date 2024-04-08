import numpy as np
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn.functional as F

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