import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

from u_datasets import OxfordPetsDataset, read_data

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert the PIL Image to a tensor
])

# Define your DataLoader
data = read_data('data/oxford/annotations/test.txt')

oxford_test_dataset = OxfordPetsDataset('data/oxford', data, transform=transform)

dataloader = DataLoader(oxford_test_dataset, batch_size=1, shuffle=True)

# Transformation pipeline with ToTensor conversion
elastic_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ElasticTransform(),
    transforms.ToTensor()  # Converts PIL Image to Tensor
])

solarize_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomSolarize(threshold=128),
    transforms.ToTensor()  # Converts PIL Image to Tensor
])

posterize_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomPosterize(bits=2),
    transforms.ToTensor()  # Converts PIL Image to Tensor
])

simple_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()  # Converts PIL Image to Tensor
])


# Process a single batch from the dataloader
for batch in dataloader:
    original_images = batch[0]  # Assuming images are the first element of the batch
    batch_size = original_images.size(0)

    print(original_images.shape)

    # Create a list to hold the processed images
    grid_images = []

    # Process each image in the batch
    for img_tensor in original_images:
        # Normalize image tensor to [0, 1] for displaying if not already
        img = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

        img = F.to_pil_image(img)

        # Apply the transformations
        img_elastic = elastic_transform(img)
        img_solarize = solarize_transform(img)
        img_posterize = posterize_transform(img)

        img = simple_transform(img)

        # Add the images to the list for the grid
        grid_images.extend([img, img_elastic, img_solarize, img_posterize])

    # Convert the list of tensors to a grid
    grid = make_grid(grid_images, nrow=batch_size)

    # Convert grid to numpy array and transpose axes to (H, W, C) for plt.imshow
    np_grid = grid.numpy().transpose((1, 2, 0))

    # Display the grid
    plt.figure(figsize=(10, 10))
    plt.imshow(np_grid, interpolation='nearest')
    plt.axis('off')
    plt.savefig('tr_vis.png')
    plt.show()

    break  # Show only the first batch