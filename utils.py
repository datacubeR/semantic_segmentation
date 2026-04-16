import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image


def show_image_and_mask_from_tensor(
    data,
    image_id,
    figsize=(10, 5),
    image_label="image",
    mask_label="mask",
    format="int64",
):
    image = data[image_id][image_label].permute(1, 2, 0).numpy().astype(format)
    mask = data[image_id][mask_label].numpy().astype(format)
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(image)
    ax[0].axis("off")
    ax[1].imshow(mask)
    ax[1].axis("off")


def show_image_and_mask_from_pil(image_path, mask_path, figsize=(10, 5)):
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(image)
    ax[0].axis("off")
    ax[1].imshow(mask)
    ax[1].axis("off")


def show_image_and_mask_from_rasterio(image_path, mask_path, figsize=(10, 5)):
    image = rasterio.open(image_path).read().transpose(1, 2, 0)
    mask = rasterio.open(mask_path).read().transpose(1, 2, 0)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(image)
    ax[0].axis("off")
    ax[1].imshow(mask)
    ax[1].axis("off")
