import argparse
import glob

from rich import print

from splitters.split_utils import create_split_folders, split_images_and_masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into training and validation sets."
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
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the validation set. Default is 0.2 (20%).",
    )
    args = parser.parse_args()

    try:
        IMAGE_GLOB = f"{args.dataset_folder}/{args.image_folder}"
        MASK_GLOB = f"{args.dataset_folder}/{args.mask_folder}"
        TRAIN_FOLDER = f"{args.dataset_folder}/train"
        VAL_FOLDER = f"{args.dataset_folder}/val"

        train_images, val_images, train_masks, val_masks = split_images_and_masks(
            IMAGE_GLOB, MASK_GLOB, test_size=args.test_size, random_state=42
        )

        create_split_folders(TRAIN_FOLDER, images=train_images, masks=train_masks)
        create_split_folders(VAL_FOLDER, images=val_images, masks=val_masks)
        images_folder = glob.glob(f"{TRAIN_FOLDER}/images/*")
        mask_folder = glob.glob(f"{TRAIN_FOLDER}/masks/*")

        print("✅ [bold green]Dataset split completed successfully![/bold green]")
        print(
            f"[bold green]{len(images_folder)} training images created in {TRAIN_FOLDER}/images[/bold green]"
        )
        print(
            f"[bold green]{len(mask_folder)} training masks created in {TRAIN_FOLDER}/masks[/bold green]"
        )
        print(
            f"[bold blue]{len(glob.glob(f'{VAL_FOLDER}/images/*'))} validation images created in {VAL_FOLDER}/images[/bold blue]"
        )
        print(
            f"[bold blue]{len(glob.glob(f'{VAL_FOLDER}/masks/*'))} validation masks created in {VAL_FOLDER}/masks[/bold blue]"
        )
    except Exception as e:
        print(f"❌ [bold red]Error during dataset split: {e}[/bold red]")
        print(f"[red]{e}[/red]")
