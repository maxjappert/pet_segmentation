import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from u_datasets import read_data, OxfordPetsDataset
from u_models import SimCLR, SegmentationHead


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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.ToTensor(),  # Convert the PIL Image to a tensor
    ])

    model = SimCLR(out_features=128).to(device)
    model.head = SegmentationHead(in_features=512, output_dim=3)
    model.load_state_dict(torch.load('finished_model.pth', map_location=device))
    model = model.to(device)

    # Update the model's forward method
    model.flatten = False

    data = read_data('data/oxford/annotations/test.txt')

    oxford_test_dataset = OxfordPetsDataset('data/oxford', data, transform=transform)

    oxford_test_dataloader = DataLoader(oxford_test_dataset, batch_size=64, shuffle=True, num_workers=12)

    model.eval()
    accuracy, iou = evaluate_model(model, oxford_test_dataloader, torch.device('cuda'))

    print(iou)
