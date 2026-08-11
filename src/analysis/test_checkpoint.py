import lightning as L
import torch
import yaml

from ..config import TrainConfig
from ..datamodules import HFDataModule, SimpleDataModule
from ..models import get_model
from ..segmentors import GridSegmentor, SimpleSegmentor

L.seed_everything(42)


def test_gridsegmentor_checkpoint(checkpoint_to_validate: str, config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cfg = TrainConfig(**config)
    model = get_model(cfg.model, **cfg.model_kwargs.model_dump())

    train_path = f"{cfg.dataset_name}_HF/{cfg.dataset_name}_train_patches-{cfg.patch_size}x{cfg.patch_size}/*"
    val_path = f"{cfg.dataset_name}_HF/{cfg.dataset_name}_validation/*"
    test_path = f"{cfg.dataset_name}_HF/{cfg.dataset_name}_test/*"

    segmentation_model = GridSegmentor.load_from_checkpoint(
        checkpoint_to_validate,
        model=model,
        n_classes=cfg.n_classes,
        criterion=cfg.get_loss(),
        hf_model=cfg.model,
        batch_size=cfg.batch_size,
        patch_size=cfg.patch_size,
        overlap=cfg.overlap,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        weights_only=False,
    )

    dm = HFDataModule(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=cfg.batch_size,
    )

    torch.set_float32_matmul_precision("high")
    trainer = L.Trainer(
        enable_model_summary=True,
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=True,
        accumulate_grad_batches=cfg.grad_accumulation_batches,
        log_every_n_steps=1,
        precision=cfg.precision,
        logger=False,
    )

    return trainer, segmentation_model, dm


def test_simplesegmentor_checkpoint(checkpoint_to_validate: str, config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cfg = TrainConfig(**config)
    model = get_model(cfg.model, **cfg.model_kwargs.model_dump())

    segmentation_model = SimpleSegmentor.load_from_checkpoint(
        checkpoint_to_validate,
        model=model,
        n_classes=cfg.n_classes,
        criterion=cfg.get_loss(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        hf_model=cfg.model,
        weights_only=False,
    )

    dm = SimpleDataModule(
        dataset_name=cfg.dataset_name,
        batch_size=cfg.batch_size,
        val_test_batch_size=cfg.val_test_batch_size,
    )

    torch.set_float32_matmul_precision("high")
    trainer = L.Trainer(
        enable_model_summary=True,
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=True,
        accumulate_grad_batches=cfg.grad_accumulation_batches,
        log_every_n_steps=1,
        precision=cfg.precision,
        logger=False,
    )

    return trainer, segmentation_model, dm
