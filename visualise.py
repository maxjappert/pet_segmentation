import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from u_datasets import read_data, split_data, OxfordPetsDataset
from u_models import SimCLR, SegmentationHead


def visualize_segmentation(images, masks, num_images=5):
    """
    Visualizes a batch of original images and their corresponding segmentation masks.

    Parameters:
        images (Tensor): The original images.
        masks (Tensor): The predicted segmentation masks.
        num_images (int): Number of images to visualize.
    """
    # Ensure we're working with numpy arrays.
    images = images.numpy()
    masks = masks.numpy()

    fig, axs = plt.subplots(nrows=num_images, ncols=2, figsize=(10, 2 * num_images))

    for i in range(num_images):
        img = np.transpose(images[i], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        mask = masks[i][0] if masks[i].shape[0] == 1 else masks[i]  # Handle single-channel mask

        # Normalize the image if necessary
        img = (img - img.min()) / (img.max() - img.min())

        axs[i, 0].imshow(img)
        axs[i, 1].imshow(mask, cmap='gray')

        axs[i, 0].set_title('Original Image')
        axs[i, 1].set_title('Segmentation Mask')

        axs[i, 0].axis('off')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming `original_images` and `predicted_masks` are your tensors of images and masks respectively
# visualize_segmentation(original_images, predicted_masks, num_images=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

segmentation_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert the PIL Image to a tensor
])

model = SimCLR(out_features=128).to(device)
model.load_state_dict(torch.load('path_to_your_saved_model.pth', map_location=device))
model.head = SegmentationHead(in_features=512, output_dim=3)

# Update the model's forward method
model.flatten = False

data = read_data('data/oxford/annotations/test.txt')


oxford_test_dataset = OxfordPetsDataset('data/oxford', data, transform=segmentation_transform)

oxford_test_dataloader = DataLoader(oxford_test_dataset, batch_size=4, shuffle=True, num_workers=4)

images, labels = images.to(device), labels.to(device)
outputs = model(images)