import argparse
import glob
from pathlib import Path

from rich import print
from sklearn.model_selection import train_test_split
from torchgeo.datasets import LoveDA

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
    args = parser.parse_args()

    try:
        DATASET_NAME = args.dataset_folder
        TRAIN_FOLDER = f"{DATASET_NAME}/train"
        VAL_FOLDER = f"{DATASET_NAME}/val"
        TEST_FOLDER = f"{DATASET_NAME}/test"

        train_data = LoveDA(
            root=DATASET_NAME,
            split="train",
            scene=["urban", "rural"],
            download=True,
        )
        dev_data = LoveDA(
            root=DATASET_NAME,
            split="val",
            scene=["urban", "rural"],
            download=True,
        )
        train_images = sorted(
            [Path(p) for p in glob.glob(f"{DATASET_NAME}/Train/*/images_png/*")]
        )
        train_masks = sorted(
            [Path(p) for p in glob.glob(f"{DATASET_NAME}/Train/*/masks_png/*")]
        )

        dev_images = sorted(
            [Path(p) for p in glob.glob(f"{DATASET_NAME}/Val/*/images_png/*")]
        )
        dev_masks = sorted(
            [Path(p) for p in glob.glob(f"{DATASET_NAME}/Val/*/masks_png/*")]
        )

        val_images, test_images, val_masks, test_masks = train_test_split(
            dev_images, dev_masks, test_size=0.3, random_state=42, shuffle=True
        )

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
