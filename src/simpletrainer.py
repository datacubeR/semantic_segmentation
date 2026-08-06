import argparse
import os
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import ValidationError
from rich import print
from torchinfo import summary

from .config import TrainConfig
from .datamodules import SimpleDataModule
from .models import get_model
from .notify import notify
from .segmentors import SimpleSegmentor
from .system_callback import SystemMetricsCallback
from .timing_callback import TimingCallback

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    choices=["DeadTrees", "LoveDA", "deadtrees", "loveda"],
    required=True,
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["segnet", "unet", "unetpp", "upernet", "segformer", "swin", "dpt", "deeplab"],
)
parser.add_argument("--version", type=int, required=True)
args = parser.parse_args()

# ===========================================
# Configuration Parameters
# ===========================================
os.system("cls" if os.name == "nt" else "clear")
print("[bold yellow]Starting Simple Training...[/bold yellow]")

config_path = f"config_files/v{args.version}/{args.model.lower()}_{args.dataset.lower()}_v{args.version}.yaml"

if not Path(config_path).exists():
    raise FileNotFoundError("Config file not found")
else:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    VERSION = f"v{args.version}"
    print(
        f"[bold green]Config file loaded successfully from {Path(config_path).name}[/bold green]"
    )

try:
    cfg = TrainConfig(**config)
except ValidationError as e:
    notify(
        f"❌ Invalid Configuration for {Path(config_path).name}.",
        title="Training Failed",
        priority="5",
    )
    print(e)
    sys.exit(1)

L.seed_everything(42)

print("[bold magenta]Initializing Model...[/bold magenta]")
model = get_model(cfg.model, **cfg.model_kwargs.model_dump())

m_summary = str(
    summary(
        model,
        input_size=(1, cfg.in_channels, cfg.image_size, cfg.image_size),
        verbose=0,
    )
)
print(f"[bold blue]{m_summary}[/bold blue]")
time.sleep(3)

if config["debug"]:
    print("[bold cyan]Debug Mode...[/bold cyan]")
    sys.exit(0)

print("[bold magenta]Initializing Data Module...[/bold magenta]")

dirpath = Path(f"l_checkpoints/{cfg.dataset_name}/{cfg.model_name}/{VERSION}")

iou_checkpoint_callback = ModelCheckpoint(
    dirpath=dirpath,
    filename="best_miou_{val_metrics/miou:.4f}",
    save_top_k=cfg.save_top_k,
    monitor="val_metrics/miou",
    mode="max",
    save_last=True,
    enable_version_counter=False,
)

f1_checkpoint_callback = ModelCheckpoint(
    dirpath=dirpath,
    filename="best_f1_{val_metrics/f1:.4f}",
    monitor="val_metrics/f1",
    mode="max",
    save_top_k=cfg.save_top_k,
    save_last=True,
    enable_version_counter=False,
)

precision_checkpoint_callback = ModelCheckpoint(
    dirpath=dirpath,
    filename="best_precision_{val_metrics/precision:.4f}",
    save_top_k=cfg.save_top_k,
    monitor="val_metrics/precision",
    mode="max",
    save_last=True,
    enable_version_counter=False,
)

recall_checkpoint_callback = ModelCheckpoint(
    dirpath=dirpath,
    filename="best_recall_{val_metrics/recall:.4f}",
    save_top_k=cfg.save_top_k,
    monitor="val_metrics/recall",
    mode="max",
    save_last=True,
    enable_version_counter=False,
)

accuracy_checkpoint_callback = ModelCheckpoint(
    dirpath=dirpath,
    filename="best_accuracy_{val_metrics/accuracy:.4f}",
    save_top_k=cfg.save_top_k,
    monitor="val_metrics/accuracy",
    mode="max",
    save_last=True,
    enable_version_counter=False,
)

logger = TensorBoardLogger(
    save_dir="tb_logs",
    version=VERSION,
    name=f"{cfg.dataset_name}_{cfg.model_name}",
    default_hp_metric=False,
)

system_metrics_callback = SystemMetricsCallback()
timing_callback = TimingCallback()

segmentation_model = SimpleSegmentor(
    model,
    n_classes=cfg.n_classes,
    criterion=cfg.get_loss(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
    hf_model=cfg.model,
)

dm = SimpleDataModule(
    dataset_name=cfg.dataset_name,
    batch_size=cfg.batch_size,
    val_test_batch_size=cfg.val_test_batch_size,
)

print("[bold red]Starting Training...[/bold red]")
torch.set_float32_matmul_precision("high")
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

        if cfg.checkpoint_path is None and dirpath.exists():
            shutil.rmtree(dirpath)
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
            title=f"{cfg.model_name}_{cfg.dataset_name}_{VERSION} - Training Completed",
            priority="5",
        )

    except Exception:
        error_msg = traceback.format_exc()
        print(f"[bold red]Error during training:\n{error_msg}[/bold red]")
        notify(
            "❌ Go to the Computer and check the logs for more information.",
            title=f"{cfg.model_name}_{cfg.dataset_name}_{VERSION} - Training Failed",
            priority="5",
        )
