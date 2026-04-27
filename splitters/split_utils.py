import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


def split_images_and_masks(image_glob, mask_glob, test_size=0.2, random_state=42):
    image_paths = sorted(list(Path(image_glob).glob("*.tif")))
    mask_paths = sorted(list(Path(mask_glob).glob("*.tif")))

    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=random_state
    )
    return train_images, val_images, train_masks, val_masks


def create_split_folders(folder_name, images, masks):
    Path(f"{folder_name}/images").mkdir(parents=True, exist_ok=True)
    Path(f"{folder_name}/masks").mkdir(parents=True, exist_ok=True)

    for img, msk in zip(images, masks):
        destination_img = Path(folder_name) / "images" / img.name
        destination_msk = Path(folder_name) / "masks" / msk.name
        shutil.move(img, destination_img)
        shutil.move(msk, destination_msk)
