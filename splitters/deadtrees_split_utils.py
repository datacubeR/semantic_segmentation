from pathlib import Path

import rasterio
from rich import print
from sklearn.model_selection import train_test_split


def detect_non_border_images(
    image_list: list[Path], percentage_threshold: float = 0.1
) -> list[str]:
    blank_image_ids = []
    for path in image_list:
        image = rasterio.open(str(path)).read().transpose(1, 2, 0)
        ## Detecting Images with any number of white Pixels
        if (image.sum(axis=2) == 3 * 255).sum() / image[
            :, :, 0
        ].size <= percentage_threshold:
            blank_image_ids.append(Path(path).name)

    print(
        f"[bold yellow] Number of Non-Border Images: {len(blank_image_ids)} [/bold yellow]"
    )
    return blank_image_ids


def calculate_non_empty_masks(values: list[str], mask_path: Path) -> list[str]:
    valid_masks = []
    for value in values:
        if rasterio.open(mask_path / value).read().sum() > 0:
            valid_masks.append(value)

    print(f"[bold yellow] Number of non-empty masks: {len(valid_masks)} [/bold yellow]")
    return valid_masks


def deadtrees_split_images_and_masks(
    non_border_image_names: list[str],
    non_empty_mask_names: list[str],
    dev_size: float = 0.2,
    test_size: float = 0.5,
    random_state: int = 42,
):

    _, dev_image_names = train_test_split(
        non_empty_mask_names, test_size=dev_size, random_state=random_state
    )

    train_image_names = list(set(non_border_image_names) - set(dev_image_names))
    val_image_names, test_image_names = train_test_split(
        dev_image_names, test_size=test_size, random_state=random_state
    )

    return train_image_names, val_image_names, test_image_names


def create_image_paths(image_names: list[str], image_path: Path) -> list[Path]:
    paths = []
    for image_name in image_names:
        paths.append(image_path / image_name)
    return paths
