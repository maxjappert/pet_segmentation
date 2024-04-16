import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from utils.u_datasets import read_data, split_data, OxfordPetsDataset
from utils.u_models import SimCLR, SegmentationHead

np.set_printoptions(precision=3)

# For reproducibility
seed = 48
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def visualize_segmentation(images, masks, labels, num_images=5, name='pretrained'):
    """
    Visualizes a batch of original images and their corresponding segmentation masks.

    Parameters:
        images (Tensor): The original images.
        masks (Tensor): The predicted segmentation masks.
        num_images (int): Number of images to visualize.
    """
    # Ensure we're working with numpy arrays.
    images = images.numpy()#.squeeze()
    masks = torch.argmax(masks, dim=1).numpy()#.squeeze()
    labels = labels.numpy()#.squeeze()

    fig, axs = plt.subplots(nrows=num_images, ncols=3, figsize=(10, 3 * num_images))

    for i in range(num_images):
        img = images[i].transpose((1, 2, 0)) #np.transpose(images[i], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        mask = masks[i]#.transpose((1, 2, 0))  # Handle single-channel mask
        label = labels[i]

        print(masks.min())
        print(masks.max())

        # Normalize the image if necessary
        img = (img - img.min()) / (img.max() - img.min())
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        label = (label - label.min()) / (label.max() - label.min())

        axs[i, 0].imshow(img)
        axs[i, 1].imshow(mask, cmap='gray')
        axs[i, 2].imshow(label, cmap='gray')

        axs[i, 0].set_title('Original Image')
        axs[i, 1].set_title('Learned Mask')
        axs[i, 2].set_title('Ground Truth Mask')

        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f'masks_{name}.png')
    plt.show()

# Example usage:
# Assuming ⁠ original_images ⁠ and ⁠ predicted_masks ⁠ are your tensors of images and masks respectively
# visualize_segmentation(original_images, predicted_masks, num_images=5)

def vis(images, labels, pretrained=True):
    device = torch.device("cpu")

    model = SimCLR(out_features=128).to(device)
    model.head = SegmentationHead(in_features=512, output_dim=3)
    model.load_state_dict(torch.load('finished_model.pth' if pretrained else 'benchmark.pth', map_location=device))
    model = model.to(device)

    # Update the model's forward method
    model.flatten = False

    model.eval()
    with torch.no_grad():
        outputs = model(images)

    visualize_segmentation(images.detach().cpu(), outputs.detach().cpu(), labels.detach().cpu(), num_images=num_images, name='pretrained' if pretrained else 'benchmark')


data = read_data('data/oxford/annotations/test.txt')

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert the PIL Image to a tensor
])

oxford_test_dataset = OxfordPetsDataset('data/oxford', data, transform=transform)

num_images = 4
oxford_test_dataloader = DataLoader(oxford_test_dataset, batch_size=num_images, shuffle=True)

images, labels = next(iter(oxford_test_dataloader))

vis(images, labels, pretrained=True)
vis(images, labels, pretrained=False)
