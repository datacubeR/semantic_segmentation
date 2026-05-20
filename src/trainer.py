import os
import time
import warnings
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from rich import print
from torchinfo import summary

from .datamodules import HFDataModule
from .models import SegNet
from .segmentators import Segmentator

warnings.filterwarnings("ignore")
## TODO: Add Config Files
## TODO: Add Augmentations
## TODO: Image Logger
# ===========================================
# User Defined Parameters
# ===========================================
CHECKPOINT_PATH = None
MODEL_NAME = "segnet_v1"
DATASET_NAME = "Vaihingen_HF"
PATCH_SIZE = 256
IN_CHANNELS = 3
N_CLASSES = 6
BATCH_SIZE = 8
MAX_EPOCHS = 1
GRAD_ACCUMULATION_BATCHES = 32
PRECISION = "16-mixed"


train_path = f"{DATASET_NAME}/Vaihingen_train_patches-256x256/*"
val_path = f"{DATASET_NAME}/Vaihingen_validation/*"

print("[bold magenta]Initializing Model and Data Module...[/bold magenta]")

L.seed_everything(42)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"l_checkpoints/{DATASET_NAME}/{MODEL_NAME}/",
    filename="{epoch:02d}-{step}-{val_iou:.3f}-{val_f1:.3f}-{val_loss:.3f}",
    save_top_k=1,
    monitor="val_iou",
    mode="max",
    save_last=True,
)

logger = TensorBoardLogger(
    save_dir="tb_logs", version=MODEL_NAME, name=DATASET_NAME, default_hp_metric=False
)


model = SegNet(in_channels=IN_CHANNELS, out_channels=N_CLASSES)

time.sleep(3)
os.system("cls" if os.name == "nt" else "clear")
m_summary = str(
    summary(model, input_size=(1, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE), verbose=0)
)
print(f"[bold blue]{m_summary}[/bold blue]")

segmentation_model = Segmentator(model, n_classes=N_CLASSES, batch_size=BATCH_SIZE)
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
    if CHECKPOINT_PATH is not None and not Path(CHECKPOINT_PATH).exists():
        raise RuntimeError("Checkpoint path does not exist")

    trainer.fit(segmentation_model, datamodule=dm, ckpt_path=CHECKPOINT_PATH)

    trainer.logger.log_hyperparams(
        segmentation_model.hparams,
        {
            "hp_metric": trainer.callback_metrics["val_iou"].item(),
        },
    )
    print("[bold green]Training Completed![/bold green]")
