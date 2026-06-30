import warnings

import lightning as L
import matplotlib.pyplot as plt
import yaml

from src.config import TrainConfig
from src.datamodules import HFDataModule
from src.models import get_model
from src.segmentors import GridSegmentor

warnings.filterwarnings("ignore")

warnings.filterwarnings(
    "ignore",
    message="You called `self.log",
)

L.seed_everything(42)


def plot_comparison(model_output):
    for idx, image in enumerate(model_output, start=1):
        pred = image[0].permute(1, 2, 0).numpy()
        gt = image[1].permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(pred)
        axes[0].set_title("Prediction")
        axes[0].axis("off")

        axes[1].imshow(gt)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        fig.suptitle(f"Image {idx}", fontsize=16)

        plt.tight_layout()
        plt.show()


def test_checkpoint(checkpoint_to_validate: str, config_path: str):
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