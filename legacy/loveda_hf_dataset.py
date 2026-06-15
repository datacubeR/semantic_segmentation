import os
import warnings

from rasterio.errors import NotGeoreferencedWarning
from rich import print
from torchgeo.datasets import LoveDA

from .datamodules.data_classes import HFDataset

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ===========================================
# Parameters
# ===========================================
GENERATE_VALIDATION = False
PATCH_SIZE = 256
OVERLAP = 0
TRAIN_SHARD_SIZE = 3000
VAL_SHARD_SIZE = 2


# ===========================================
# Glob Patterns
# ===========================================
ROOT = "loveda_dataset"

# ===========================================
# Output Directory
# ===========================================
DATASET_NAME = "LoveDA"
OUTPUT_DIR = f"{DATASET_NAME}_HF"


# ===========================================
# Source Code
# ===========================================

os.system("cls" if os.name == "nt" else "clear")

training_output_path = (
    f"{OUTPUT_DIR}/{DATASET_NAME}_train_patches-{PATCH_SIZE}x{PATCH_SIZE}"
)
validation_output_path = f"{OUTPUT_DIR}/{DATASET_NAME}_validation"

print(f"[bold magenta]Loading {DATASET_NAME} Dataset...[/bold magenta]")

train_data = LoveDA(
    root=ROOT,
    split="train",
    scene=["urban", "rural"],
)

val_data = LoveDA(
    root=ROOT,
    split="val",
    scene=["urban", "rural"],
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
    patch_kwargs=dict(patch_size=(PATCH_SIZE, PATCH_SIZE), overlap=(OVERLAP, OVERLAP)),
    shard_size=TRAIN_SHARD_SIZE,
)


if GENERATE_VALIDATION:
    print("[bold blue]Generating Validation Tensors...[/bold blue]")
    val_hf_dataset = HFDataset(val_data)
    val_hf_dataset.build_full(
        validation_output_path,
        shard_size=VAL_SHARD_SIZE,
    )

print("[bold green]Done![/bold green]")
