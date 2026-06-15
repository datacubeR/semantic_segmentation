import time

import lightning as L


class TimingCallback(L.Callback):
    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):

        epoch_time = time.perf_counter() - self.epoch_start_time

        pl_module.log(
            "timing/epoch_seconds",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_train_end(self, trainer, pl_module):

        total_train_time = time.perf_counter() - self.train_start_time

        trainer.logger.experiment.add_scalar(
            "timing/total_training_seconds",
            total_train_time,
            trainer.global_step,
        )
