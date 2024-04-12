import numpy as np
import torch
from config import *


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Returns the accuracy and Intersection over Union
    """
    model.eval()
    intersection_total, union_total = 0, 0
    pixel_correct, pixel_count = 0, 0

    for data in dataloader:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)
        targets = torch.squeeze(labels)

        intersection_total += torch.logical_and(preds, targets).sum()
        union_total += torch.logical_or(preds, targets).sum()

        pixel_correct += (preds == targets).sum()
        pixel_count += targets.numel()

    iou = (intersection_total / union_total).item()
    accuracy = (pixel_correct / pixel_count).item()

    return accuracy, iou
