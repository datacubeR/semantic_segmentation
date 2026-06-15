import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


def create_split_folders(folder_name, images, masks):
    Path(f"{folder_name}/images").mkdir(parents=True, exist_ok=True)
    Path(f"{folder_name}/masks").mkdir(parents=True, exist_ok=True)

    for img, msk in zip(images, masks):
        destination_img = Path(folder_name) / "images" / img.name
        destination_msk = Path(folder_name) / "masks" / msk.name
        shutil.move(img, destination_img)
        shutil.move(msk, destination_msk)


def split_images_and_masks(
    image_glob, mask_glob, train_size=0.8, test_size=0.2, random_state=42
):
    image_paths = sorted(list(Path(image_glob).glob("*.tif")))
    mask_paths = sorted(list(Path(mask_glob).glob("*.tif")))

    train_images, dev_images, train_masks, dev_masks = train_test_split(
        image_paths, mask_paths, train_size=train_size, random_state=random_state
    )

    val_images, test_images, val_masks, test_masks = train_test_split(
        dev_images, dev_masks, test_size=test_size, random_state=random_state
    )

    return train_images, val_images, test_images, train_masks, val_masks, test_masks
