import argparse
import os
import warnings

from rasterio.errors import NotGeoreferencedWarning
from rich import print

from .datamodules.data_classes import HFDataset, PotsdamVaihingen

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


# ===========================================
# Source Code
# ===========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Dataset into a Hugging Face Dataset to improve loading and training performance."
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        help="Folder name where the dataset is stored. This folder should contain the image and mask folders.",
        required=True,
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        help="Size of the patches to generate.",
        default=256,
    )
    parser.add_argument(
        "--train-shard-size",
        type=int,
        help="Number of samples per shard for the training set.",
        default=5000,
    )
    parser.add_argument(
        "--val-shard-size",
        type=int,
        help="Number of samples per shard for the validation set.",
        default=10,
    )
    parser.add_argument(
        "--test-shard-size",
        type=int,
        help="Number of samples per shard for the test set.",
        default=10,
    )

    parser.add_argument(
        "-tr",
        "--training",
        help="Whether to generate the training dataset.",
        action="store_true",
    )

    parser.add_argument(
        "-vt",
        "--validation-test",
        help="Whether to generate validation and test datasets.",
        action="store_true",
    )

    args = parser.parse_args()
    # ===========================================
    # Parameters
    # ===========================================
    DATASET_NAME = args.dataset_folder
    OUTPUT_DIR = f"{DATASET_NAME}_HF"
    PATCH_SIZE = args.patch_size
    OVERLAP = 0
    TRAIN_SHARD_SIZE = args.train_shard_size
    VAL_SHARD_SIZE = args.val_shard_size
    TEST_SHARD_SIZE = args.test_shard_size

    # ===========================================
    # Glob Patterns
    # ===========================================
    TRAIN_VAIHINGEN_IMAGE_GLOB = f"{DATASET_NAME}_dataset/train/images/*.tif"
    TRAIN_VAIHINGEN_MASK_GLOB = f"{DATASET_NAME}_dataset/train/masks/*.tif"
    VAL_VAIHINGEN_IMAGE_GLOB = f"{DATASET_NAME}_dataset/val/images/*.tif"
    VAL_VAIHINGEN_MASK_GLOB = f"{DATASET_NAME}_dataset/val/masks/*.tif"
    TEST_VAIHINGEN_IMAGE_GLOB = f"{DATASET_NAME}_dataset/test/images/*.tif"
    TEST_VAIHINGEN_MASK_GLOB = f"{DATASET_NAME}_dataset/test/masks/*.tif"

    os.system("cls" if os.name == "nt" else "clear")

    if args.training:
        print("Generating Training Dataset...")
        training_output_path = (
            f"{OUTPUT_DIR}/{DATASET_NAME}_train_patches-{PATCH_SIZE}x{PATCH_SIZE}"
        )
        print(f"[bold magenta]Loading {DATASET_NAME} Dataset...[/bold magenta]")

        train_data = PotsdamVaihingen(
            image_glob=TRAIN_VAIHINGEN_IMAGE_GLOB,
            mask_glob=TRAIN_VAIHINGEN_MASK_GLOB,
            reduce_mask=True,
        )

        print(
            f"[bold yellow]Number of Original Training Images: {len(train_data)}[/bold yellow]"
        )

        print("[bold blue]Generating Training Patches...[/bold blue]")
        train_hf_dataset = HFDataset(train_data)
        train_hf_dataset.build_patches(
            training_output_path,
            patch_kwargs=dict(
                patch_size=(PATCH_SIZE, PATCH_SIZE), overlap=(OVERLAP, OVERLAP)
            ),
            shard_size=TRAIN_SHARD_SIZE,
        )

    if args.validation_test:
        print("Generating Validation and Test Datasets...")
        validation_output_path = f"{OUTPUT_DIR}/{DATASET_NAME}_validation"
        test_output_path = f"{OUTPUT_DIR}/{DATASET_NAME}_test"

        val_data = PotsdamVaihingen(
            image_glob=VAL_VAIHINGEN_IMAGE_GLOB,
            mask_glob=VAL_VAIHINGEN_MASK_GLOB,
            reduce_mask=True,
        )

        test_data = PotsdamVaihingen(
            image_glob=TEST_VAIHINGEN_IMAGE_GLOB,
            mask_glob=TEST_VAIHINGEN_MASK_GLOB,
            reduce_mask=True,
        )

        print(
            f"[bold yellow]Number of Original Validation Images: {len(val_data)}[/bold yellow]"
        )
        print(
            f"[bold yellow]Number of Original Test Images: {len(test_data)}[/bold yellow]"
        )

        print("[bold blue]Generating Validation Tensors...[/bold blue]")

        val_hf_dataset = HFDataset(val_data)
        val_hf_dataset.build_full(validation_output_path, shard_size=VAL_SHARD_SIZE)

        print("[bold blue]Generating Test Tensors...[/bold blue]")

        test_hf_dataset = HFDataset(test_data)
        test_hf_dataset.build_full(test_output_path, shard_size=TEST_SHARD_SIZE)

    print("[bold green]Done![/bold green]")
