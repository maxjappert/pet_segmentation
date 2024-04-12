import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from PIL import Image, ImageOps

def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                  boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

batch_file_path = "data/imagenet64/train_data_batch_1"
with open(batch_file_path, 'rb') as f:
    data = pickle.load(f, encoding='bytes')

images = data['data']
img = images[3]
reshaped = np.array(img).reshape((3, 64, 64)).transpose((1, 2, 0))
original_image = Image.fromarray(reshaped).convert('RGB')
plot([img],row_title=["Original Image"])

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

posterize_bits = 4
solarize_threshold = 192

transform_elastic = transforms.Compose([
    transforms.ElasticTransform(),
    transforms.ToTensor()
])
transform_solarize = transforms.Compose([
    transforms.RandomSolarize(threshold=solarize_threshold),
    transforms.ToTensor()
])
transform_posterize = transforms.Compose([
    transforms.RandomPosterize(bits=posterize_bits),
    transforms.ToTensor()
])
transform_elastic_solarize = transforms.Compose([
    transforms.ElasticTransform(),
    transforms.RandomSolarize(threshold=solarize_threshold),
    transforms.ToTensor()
])
transform_elastic_posterize = transforms.Compose([
    transforms.ElasticTransform(),
    transforms.RandomPosterize(bits=posterize_bits),
    transforms.ToTensor()
])
transform_solarize_posterize = transforms.Compose([
    transforms.RandomPosterize(bits=posterize_bits),
    transforms.RandomSolarize(threshold=solarize_threshold),
    transforms.ToTensor()
])
transform_elastic_solarize_posterize = transforms.Compose([
    transforms.ElasticTransform(),
    transforms.RandomPosterize(bits=posterize_bits),
    transforms.RandomSolarize(threshold=solarize_threshold),
    transforms.ToTensor()
])

transformations = [
    transform_elastic,
    transform_solarize,
    transform_posterize,
    transform_elastic_solarize,
    transform_elastic_posterize,
    transform_solarize_posterize,
    transform_elastic_solarize_posterize
]

transformation_names = ["Elastic", "Solarize", "Posterize", 
                        "El. + Sol.", "El. + Pos.",
                        "Sol. + Pos.", "El. + Sol. + Pos."]

for i, transform in enumerate(transformations):
    transformed_image = [transform(img) for _ in range(4)]
    row = [img] + transformed_image
    plot(row, row_title=[transformation_names[i]])
