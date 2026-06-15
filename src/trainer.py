import argparse
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from rich import print
from torchinfo import summary
from transformers import AutoImageProcessor

from .config import TrainConfig
from .datamodules import HFDataModule
from .models import get_model
from .notify import notify
from .segmentors import GridSegmentor
from .system_callback import SystemMetricsCallback
from .timing_callback import TimingCallback

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str)
args = parser.parse_args()

# ===========================================
# Configuration Parameters
# ===========================================
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

config_path = Path(args.config)
if not config_path.exists():
    raise FileNotFoundError("Config file not found")
else:
    print(
        f"[bold green]Config file loaded successfully from {args.config}[/bold green]"
    )
cfg = TrainConfig(**config)

L.seed_everything(42)

print("[bold magenta]Initializing Model...[/bold magenta]")
model = get_model(cfg.model, **cfg.model_kwargs.model_dump())

image_processor = None
if cfg.model_name == "mask2former":
    image_processor = AutoImageProcessor.from_pretrained(
        cfg.model_kwargs.pretrained_model_name_or_path
    )

time.sleep(3)
os.system("cls" if os.name == "nt" else "clear")
m_summary = str(
    summary(
        model,
        input_size=(1, cfg.in_channels, cfg.patch_size, cfg.patch_size),
        verbose=0,
    )
)
print(f"[bold blue]{m_summary}[/bold blue]")

if config["debug"]:
    print("[bold cyan]Debug Mode...[/bold cyan]")
    sys.exit(0)

print("[bold magenta]Initializing Data Module...[/bold magenta]")

train_path = f"{cfg.dataset_name}_HF/{cfg.dataset_name}_train_patches-{cfg.patch_size}x{cfg.patch_size}/*"
val_path = f"{cfg.dataset_name}_HF/{cfg.dataset_name}_validation/*"


iou_checkpoint_callback = ModelCheckpoint(
    dirpath=f"l_checkpoints/{cfg.dataset_name}/{cfg.model_name}/{cfg.version}",
    # filename="{epoch:02d}-{step}-{val_iou:.3f}-{val_f1:.3f}-{val_loss:.3f}",
    filename="best_iou.ckpt",
    save_top_k=1,
    monitor="metrics/val_iou",
    mode="max",
    save_last=True,
    enable_version_counter=False,
)

f1_checkpoint_callback = ModelCheckpoint(
    dirpath=f"l_checkpoints/{cfg.dataset_name}/{cfg.model_name}/{cfg.version}",
    filename="best_f1.ckpt",
    save_top_k=1,
    monitor="metrics/val_f1",
    mode="max",
    save_last=True,
    enable_version_counter=False,
)

precision_checkpoint_callback = ModelCheckpoint(
    dirpath=f"l_checkpoints/{cfg.dataset_name}/{cfg.model_name}/{cfg.version}",
    filename="best_precision.ckpt",
    save_top_k=1,
    monitor="metrics/val_precision",
    mode="max",
    save_last=True,
    enable_version_counter=False,
)

recall_checkpoint_callback = ModelCheckpoint(
    dirpath=f"l_checkpoints/{cfg.dataset_name}/{cfg.model_name}/{cfg.version}",
    filename="best_recall.ckpt",
    save_top_k=1,
    monitor="metrics/val_recall",
    mode="max",
    save_last=True,
    enable_version_counter=False,
)

accuracy_checkpoint_callback = ModelCheckpoint(
    dirpath=f"l_checkpoints/{cfg.dataset_name}/{cfg.model_name}/{cfg.version}",
    filename="best_accuracy.ckpt",
    save_top_k=1,
    monitor="metrics/val_accuracy",
    mode="max",
    save_last=True,
    enable_version_counter=False,
)

logger = TensorBoardLogger(
    save_dir="tb_logs",
    version=cfg.version,
    name=f"{cfg.dataset_name}_{cfg.model_name}",
    default_hp_metric=False,
)

system_metrics_callback = SystemMetricsCallback()
timing_callback = TimingCallback()

segmentation_model = GridSegmentor(
    model,
    n_classes=cfg.n_classes,
    criterion=cfg.get_loss(),
    image_processor=image_processor,
    hf_model=cfg.model,
    batch_size=cfg.batch_size,
    patch_size=cfg.patch_size,
    overlap=cfg.overlap,
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)
dm = HFDataModule(
    train_path=train_path,
    val_path=val_path,
    batch_size=cfg.batch_size,
)

print("[bold red]Starting Training...[/bold red]")
trainer = L.Trainer(
    enable_model_summary=True,
    max_epochs=cfg.max_epochs,
    accelerator="gpu",
    devices=1,
    enable_progress_bar=True,
    callbacks=[
        iou_checkpoint_callback,
        f1_checkpoint_callback,
        precision_checkpoint_callback,
        recall_checkpoint_callback,
        accuracy_checkpoint_callback,
        system_metrics_callback,
        timing_callback,
    ],
    accumulate_grad_batches=cfg.grad_accumulation_batches,
    log_every_n_steps=1,
    precision=cfg.precision,
    logger=logger,
)


if __name__ == "__main__":
    try:
        if cfg.checkpoint_path is not None and not Path(cfg.checkpoint_path).exists():
            raise RuntimeError("Checkpoint path does not exist")

        if cfg.checkpoint_path is not None:
            print(
                f"[bold yellow]Resuming from checkpoint: {cfg.checkpoint_path}[/bold yellow]"
            )
        else:
            print(
                "[bold yellow]No checkpoint path provided. Starting fresh training...[/bold yellow]"
            )
        start_time = time.time()
        trainer.fit(segmentation_model, datamodule=dm, ckpt_path=cfg.checkpoint_path)

        trainer.logger.log_hyperparams(
            segmentation_model.hparams,
            {
                "hp_metric": iou_checkpoint_callback.best_model_score.item(),
                "best_iou": iou_checkpoint_callback.best_model_score.item(),
                "best_f1": f1_checkpoint_callback.best_model_score.item(),
                "best_precision": precision_checkpoint_callback.best_model_score.item(),
                "best_recall": recall_checkpoint_callback.best_model_score.item(),
                "best_accuracy": accuracy_checkpoint_callback.best_model_score.item(),
            },
        )
        print("[bold green]Training Completed![/bold green]")
        end_time = time.time()
        print(
            f"[bold cyan]Training Time: {end_time - start_time:.2f} seconds[/bold cyan]"
        )
        notify(
            f"✅\n\n Training Time: {(end_time - start_time) / 60:.2f} mins. \n\n Validation IoU: {iou_checkpoint_callback.best_model_score.item():.3f}.",
            title=f"{cfg.model_name}_{cfg.dataset_name}_{cfg.version} - Training Completed",
            priority="5",
        )

    except Exception:
        error_msg = traceback.format_exc()
        print(f"[bold red]Error during training:\n{error_msg}[/bold red]")
        notify(
            "❌ Go to the Computer and check the logs for more information.",
            title=f"{cfg.model_name}_{cfg.dataset_name}_{cfg.version} - Training Failed",
            priority="5",
        )
