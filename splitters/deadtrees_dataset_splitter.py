import argparse
from pathlib import Path

from rich import print

from .deadtrees_split_utils import (
    calculate_non_empty_masks,
    create_image_paths,
    deadtrees_split_images_and_masks,
    detect_non_border_images,
)
from .split_utils import create_split_folders

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into training, validation and test sets."
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        help="Folder name where the dataset is stored. This folder should contain the image and mask folders.",
        required=True,
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        help="Folder name where the image files are stored.",
        required=True,
    )
    parser.add_argument(
        "--mask-folder",
        type=str,
        help="Folder name where the mask images are stored.",
        required=True,
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.8,
        help="Proportion of the dataset to include as Train set. Default is 0.8 (80%).",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the Validation dataset to be include as the train set. Default is 0.2 (20%).",
    )
    args = parser.parse_args()

    try:
        DATASET_NAME = args.dataset_folder
        IMAGE_FOLDER = args.image_folder
        MASK_FOLDER = args.mask_folder
        TRAIN_SIZE = args.train_size
        TEST_SIZE = args.test_size
        dev_size = 1 - TRAIN_SIZE

        IMAGE_PATH = Path(f"{DATASET_NAME}/{IMAGE_FOLDER}")
        MASK_PATH = Path(f"{DATASET_NAME}/{MASK_FOLDER}")
        TRAIN_FOLDER = f"{DATASET_NAME}/train"
        VAL_FOLDER = f"{DATASET_NAME}/val"
        TEST_FOLDER = f"{DATASET_NAME}/test"

        image_paths = sorted(IMAGE_PATH.glob("*.tif"))
        mask_paths = sorted(MASK_PATH.glob("*.tif"))

        print(
            "[bold blue] Calculating the number of images... This could take a while... [/bold blue]"
        )
        non_border_image_names = detect_non_border_images(image_paths)
        non_empty_mask_names = calculate_non_empty_masks(
            non_border_image_names, MASK_PATH
        )

        train_image_names, val_image_names, test_image_names = (
            deadtrees_split_images_and_masks(
                non_border_image_names,
                non_empty_mask_names,
                dev_size=dev_size,
                test_size=TEST_SIZE,
                random_state=42,
            )
        )

        train_images = create_image_paths(train_image_names, IMAGE_PATH)
        train_masks = create_image_paths(train_image_names, MASK_PATH)
        val_images = create_image_paths(val_image_names, IMAGE_PATH)
        val_masks = create_image_paths(val_image_names, MASK_PATH)
        test_images = create_image_paths(test_image_names, IMAGE_PATH)
        test_masks = create_image_paths(test_image_names, MASK_PATH)

        print(
            f"[bold yellow] The process with create {len(train_images)} training images, {len(val_images)} validation images, and {len(test_images)} test images.[/bold yellow]"
        )

        output = input("Do you want to proceed with moving the files? (y/n): ")

        if output == "y":
            print("[bold blue] Proceeding with file move... [/bold blue]")
            create_split_folders(TRAIN_FOLDER, images=train_images, masks=train_masks)
            create_split_folders(VAL_FOLDER, images=val_images, masks=val_masks)
            create_split_folders(TEST_FOLDER, images=test_images, masks=test_masks)
            print("[bold green] ✅ Dataset succesfully splitted! [/bold green]")
        else:
            print("[bold red] ❌ Aborting file move. [/bold red]")

    except Exception as e:
        print(f"❌ [bold red]Error during dataset split: {e}[/bold red]")
        print(f"[red]{e}[/red]")
