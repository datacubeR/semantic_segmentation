import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from rich import print
from torchinfo import summary

from .datamodules import HFDataModule
from .losses import get_loss
from .models import get_model
from .notify import notify
from .segmentators import Segmentator

warnings.filterwarnings("ignore")
## TODO: Add Augmentations *

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

config_path = Path(args.config)
if not config_path.exists():
    raise FileNotFoundError("Config file not found")
else:
    print(
        f"[bold green]Config file loaded successfully from {args.config}[/bold green]"
    )
# ===========================================
# Training Parameters
# ===========================================
CHECKPOINT_PATH = config["checkpoint_path"]
MODEL_NAME = config["model_name"]
DATASET_NAME = config["dataset_name"]
VERSION = config["version"]
IN_CHANNELS = config["in_channels"]
N_CLASSES = config["n_classes"]
PATCH_SIZE = config["patch_size"]
OVERLAP = config["overlap"]
BATCH_SIZE = config["batch_size"]
MAX_EPOCHS = config["max_epochs"]
GRAD_ACCUMULATION_BATCHES = config["grad_accumulation_batches"]
PRECISION = config["precision"]
LR = config["lr"]
WEIGHT_DECAY = config["weight_decay"]
LOSS_FUNCTION = get_loss(config["loss_function"], **config["loss_kwargs"])


L.seed_everything(42)

print("[bold magenta]Initializing Model...[/bold magenta]")
model = get_model(config["model"], **config["model_kwargs"])

time.sleep(3)
os.system("cls" if os.name == "nt" else "clear")
m_summary = str(
    summary(model, input_size=(1, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE), verbose=0)
)
print(f"[bold blue]{m_summary}[/bold blue]")

if config["debug"]:
    print("[bold cyan]Debug Mode...[/bold cyan]")
    sys.exit(0)

print("[bold magenta]Initializing Data Module...[/bold magenta]")

train_path = (
    f"{DATASET_NAME}_HF/{DATASET_NAME}_train_patches-{PATCH_SIZE}x{PATCH_SIZE}/*"
)
val_path = f"{DATASET_NAME}_HF/{DATASET_NAME}_validation/*"


checkpoint_callback = ModelCheckpoint(
    dirpath=f"l_checkpoints/{DATASET_NAME}/{MODEL_NAME}/{VERSION}",
    filename="{epoch:02d}-{step}-{val_iou:.3f}-{val_f1:.3f}-{val_loss:.3f}",
    save_top_k=1,
    monitor="val_iou",
    mode="max",
    save_last=True,
)

logger = TensorBoardLogger(
    save_dir="tb_logs",
    version=VERSION,
    name=f"{DATASET_NAME}_{MODEL_NAME}",
    default_hp_metric=False,
)


segmentation_model = Segmentator(
    model,
    n_classes=N_CLASSES,
    criterion=LOSS_FUNCTION,
    batch_size=BATCH_SIZE,
    patch_size=PATCH_SIZE,
    overlap=OVERLAP,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
dm = HFDataModule(
    train_path=train_path,
    val_path=val_path,
    batch_size=segmentation_model.hparams.batch_size,
)

print("[bold red]Starting Training...[/bold red]")
trainer = L.Trainer(
    enable_model_summary=False,
    max_epochs=MAX_EPOCHS,
    accelerator="gpu",
    devices=1,
    enable_progress_bar=True,
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=GRAD_ACCUMULATION_BATCHES,
    log_every_n_steps=1,
    precision=PRECISION,
    logger=logger,
)


if __name__ == "__main__":
    try:
        if CHECKPOINT_PATH is not None and not Path(CHECKPOINT_PATH).exists():
            raise RuntimeError("Checkpoint path does not exist")

        if CHECKPOINT_PATH is not None:
            print(
                f"[bold yellow]Resuming from checkpoint: {CHECKPOINT_PATH}[/bold yellow]"
            )
        else:
            print(
                "[bold yellow]No checkpoint path provided. Starting fresh training...[/bold yellow]"
            )
        start_time = time.time()
        trainer.fit(segmentation_model, datamodule=dm, ckpt_path=CHECKPOINT_PATH)

        trainer.logger.log_hyperparams(
            segmentation_model.hparams,
            {
                "hp_metric": trainer.callback_metrics["val_iou"].item(),
            },
        )
        print("[bold green]Training Completed![/bold green]")
        end_time = time.time()
        print(
            f"[bold cyan]Training Time: {end_time - start_time:.2f} seconds[/bold cyan]"
        )
        notify(
            f"✅\n\n Training Time: {(end_time - start_time) / 60:.2f} mins. \n\n Validation IoU: {trainer.callback_metrics['val_iou'].item():.3f}.",
            title=f"{MODEL_NAME}_{DATASET_NAME}_{VERSION} - Training Completed",
            priority="5",
        )

    except Exception as e:
        print(f"[bold red]Error during training: {e}[/bold red]")
        notify(
            "❌ Go to the Computer and check the logs for more information.",
            title=f"{MODEL_NAME}_{DATASET_NAME}_{VERSION} - Training Failed",
            priority="5",
        )
