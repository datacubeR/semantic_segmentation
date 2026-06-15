import lightning as L
import psutil
import torch


class SystemMetricsCallback(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):

        logger = trainer.logger
        step = trainer.global_step

        # =====================================================
        # GPU Metrics
        # =====================================================

        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3

            gpu_memory_reserved = torch.cuda.max_memory_reserved() / 1024**3

            logger.experiment.add_scalar(
                "system/gpu_memory_allocated_gb",
                gpu_memory_allocated,
                step,
            )

            logger.experiment.add_scalar(
                "system/gpu_memory_reserved_gb",
                gpu_memory_reserved,
                step,
            )

        # =====================================================
        # RAM Metrics
        # =====================================================

        ram = psutil.virtual_memory()

        logger.experiment.add_scalar(
            "system/ram_used_percent",
            ram.percent,
            step,
        )

        logger.experiment.add_scalar(
            "system/ram_used_gb",
            ram.used / 1024**3,
            step,
        )

        # =====================================================
        # CPU Metrics
        # =====================================================

        cpu_percent = psutil.cpu_percent()

        logger.experiment.add_scalar(
            "system/cpu_percent",
            cpu_percent,
            step,
        )
