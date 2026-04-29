import argparse
import os
import time

import kornia.augmentation as K
import lightning as L
import segmentation_models_pytorch.losses as smpl
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from rich import print
from segmentation_models_pytorch import Unet
from torchinfo import summary
from torchmetrics.segmentation import DiceScore

from .datamodules import data_classes as DM
from .segmentators import RegularTrainingSegmentator

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Model Trainer")
    argparse.add_argument(
        "--config",
        required=True,
        help="Name of the config file",
    )
    args = argparse.parse_args()

    try:
        with open(f"src/config/{args.config}.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(
            f"[bold red]Error:[/bold red] Config file not found at '{args.config}'. Please check the path."
        )
        exit(1)

    try:
        VERSION = config["version"]
        PROJECT_NAME = config["project_name"]
        MODEL_NAME = config["model_name"]

        CHECKPOINT_PATH = (
            f"checkpoints/{PROJECT_NAME}/version_{VERSION}/checkpoints/last.ckpt"
            if config.get("resume")
            else None
        )

        train_deadtrees_image_glob = config["data"]["train_image_glob"]
        train_deadtrees_mask_glob = config["data"]["train_mask_glob"]
        val_deadtrees_image_glob = config["data"]["val_image_glob"]
        val_deadtrees_mask_glob = config["data"]["val_mask_glob"]

        deadtrees_dm = DM.RegularDataModule(
            dataset_cls=getattr(DM, config["data"]["dataset_cls"]),
            train_dataset_kwargs={
                "image_glob": train_deadtrees_image_glob,
                "mask_glob": train_deadtrees_mask_glob,
                "white_threshold": config["data"]["white_threshold"],
            },
            val_dataset_kwargs={
                "image_glob": val_deadtrees_image_glob,
                "mask_glob": val_deadtrees_mask_glob,
                "white_threshold": config["data"]["white_threshold"],
            },
            loader_kwargs={
                "batch_size": config["trainer"]["batch_size"],
                "num_workers": config["trainer"]["num_workers"],
            },
        )

        model = Unet(
            config["model"]["encoder_name"],
            encoder_weights=config["model"]["encoder_weights"],
            in_channels=config["model"]["in_channels"],
            classes=config["model"]["classes"],
        )

        if config["augmentation"]["apply"]:
            horizontal_flip = config["augmentation"]["horizontal_flip"]
            vertical_flip = config["augmentation"]["vertical_flip"]
            brightness = config["augmentation"]["brightness"]
            contrast = config["augmentation"]["contrast"]
            saturation = config["augmentation"]["saturation"]
            hue = config["augmentation"]["hue"]
            gaussian_noise_mean = config["augmentation"]["gaussian_noise_mean"]
            gaussian_noise_std = config["augmentation"]["gaussian_noise_std"]
            gaussian_noise_p = config["augmentation"]["gaussian_noise_p"]

        augmentation = (
            dict(
                geom_augmentation=K.AugmentationSequential(
                    K.RandomHorizontalFlip(
                        **config["augmentation"]["horizontal_flip_kwargs"]
                    ),
                    K.RandomVerticalFlip(
                        **config["augmentation"]["vertical_flip_kwargs"]
                    ),
                    data_keys=["input", "mask"],  # imagen y máscara
                ),
                intensity_augmentation=K.AugmentationSequential(
                    K.ColorJitter(**config["augmentation"]["color_jitter_kwargs"]),
                    K.RandomGaussianNoise(
                        **config["augmentation"]["random_noise_kwargs"]
                    ),
                    data_keys=["input"],  # solo imagen
                ),
            )
            if config["augmentation"]["apply"]
            else None
        )

        segmentator = RegularTrainingSegmentator(
            model=model,
            criterion=getattr(smpl, config["architecture"]["criterion"])(
                **config["architecture"]["criterion_kwargs"]
            ),
            augmentation=augmentation,
            optimizer_cls=getattr(torch.optim, config["architecture"]["optimizer"]),
            optimizer_kwargs=config["architecture"]["optimizer_kwargs"],
            metric=DiceScore(**config["architecture"]["metrics_kwargs"]),
            add_channel_dim=config["architecture"]["add_channel_dim"],
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{PROJECT_NAME}/version_{VERSION}/checkpoints",
            filename="{epoch:02d}-{step}-{val_dice:.4f}-{val_loss:.4f}",
            save_top_k=1,  # keep best 3 models
            monitor="val_metric",  # or "val_loss"
            mode="max",  # IoU → maximize
            save_last=True,  # VERY IMPORTANT for resume
        )
        logger = TensorBoardLogger(
            save_dir="logs", version=VERSION, name=f"{PROJECT_NAME}"
        )
        trainer = L.Trainer(
            max_epochs=config["trainer"]["num_epochs"],
            logger=logger,
            callbacks=[checkpoint_callback],
            accelerator=config["trainer"]["accelerator"],
            devices=config["trainer"]["devices"],
            enable_progress_bar=True,
            deterministic=False,  ## Not possible for CE
            check_val_every_n_epoch=1,
            fast_dev_run=config["fast_dev_run"],
            overfit_batches=config["overfit_batches"],
            log_every_n_steps=1,
            precision=config["trainer"]["precision"],
            accumulate_grad_batches=config["trainer"]["accumulate_grad_batches"],
        )

        if config["fast_dev_run"]:
            print(
                "[bold red] Fast development run enabled. Training run for 1 batch only for Debugging Purposes only. [/bold red]"
            )
        else:
            os.system("cls" if os.name == "nt" else "clear")
            time.sleep(2)
            print(
                f"[bold green] {PROJECT_NAME.title()} training process started...[/bold green]"
            )
            if CHECKPOINT_PATH is not None:
                init = "Resuming"
                from_ckpt = f"checkpoint: {CHECKPOINT_PATH}."
            else:
                init = "Starting"
                from_ckpt = "scratch."

            time.sleep(2)
            print(f"[bold yellow] {init} training from {from_ckpt}...[/bold yellow]")

            time.sleep(2)
            print(f"[bold blue] Model Name: {MODEL_NAME} [/bold blue]")

            print(
                f"[bold magenta] Logging to TensorBoard at: {logger.log_dir}[/bold magenta]"
            )

            time.sleep(2)
            print("[bold cyan] Model Architecture: [/bold cyan]")

            time.sleep(1)
            summary(model)

            trainer.fit(segmentator, deadtrees_dm, ckpt_path=CHECKPOINT_PATH)
    except Exception as e:
        print(f"[bold red]Error during training setup:[/bold red] {str(e)}")
        raise
