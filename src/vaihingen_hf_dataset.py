import os
import warnings

from rasterio.errors import NotGeoreferencedWarning
from rich import print

from .datamodules.data_classes import HFDataset, PotsdamVaihingen

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ===========================================
# Parameters
# ===========================================
PATCH_SIZE = 256
OVERLAP = 0
TRAIN_SHARD_SIZE = 5000
VAL_SHARD_SIZE = 10

# ===========================================
# Glob Patterns
# ===========================================
TRAIN_VAIHINGEN_IMAGE_GLOB = "Vaihingen_dataset/train/images/*.tif"
TRAIN_VAIHINGEN_MASK_GLOB = "Vaihingen_dataset/train/masks/*.tif"
VAL_VAIHINGEN_IMAGE_GLOB = "Vaihingen_dataset/val/images/*.tif"
VAL_VAIHINGEN_MASK_GLOB = "Vaihingen_dataset/val/masks/*.tif"

# ===========================================
# Output Directory
# ===========================================
DATASET_NAME = "Vaihingen"
OUTPUT_DIR = f"{DATASET_NAME}_HF"


# ===========================================
# Source Code
# ===========================================

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    training_output_path = (
        f"{OUTPUT_DIR}/{DATASET_NAME}_train_patches-{PATCH_SIZE}x{PATCH_SIZE}"
    )
    validation_output_path = f"{OUTPUT_DIR}/{DATASET_NAME}_validation"

    print(f"[bold magenta]Loading {DATASET_NAME} Dataset...[/bold magenta]")
    train_data = PotsdamVaihingen(
        image_glob=TRAIN_VAIHINGEN_IMAGE_GLOB,
        mask_glob=TRAIN_VAIHINGEN_MASK_GLOB,
        reduce_mask=True,
    )

    val_data = PotsdamVaihingen(
        image_glob=VAL_VAIHINGEN_IMAGE_GLOB,
        mask_glob=VAL_VAIHINGEN_MASK_GLOB,
        reduce_mask=True,
    )

    print(
        f"[bold yellow]Number of Original Training Images: {len(train_data)}[/bold yellow]"
    )
    print(
        f"[bold yellow]Number of Original Validation Images: {len(val_data)}[/bold yellow]"
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

    print("[bold blue]Generating Validation Tensors...[/bold blue]")

    val_hf_dataset = HFDataset(val_data)
    val_hf_dataset.build_full(validation_output_path, shard_size=VAL_SHARD_SIZE)

    print("[bold green]Done![/bold green]")
