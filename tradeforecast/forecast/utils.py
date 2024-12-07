from pytorch_lightning.callbacks import Callback
import lightning as L

class OverrideEpochStepCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: L.Trainer, L_module: L.LightningModule):
        self._log_step_as_current_epoch(trainer, L_module)

    def on_test_epoch_end(self, trainer: L.Trainer, L_module: L.LightningModule):
        self._log_step_as_current_epoch(trainer, L_module)

    def on_validation_epoch_end(self, trainer: L.Trainer, L_module: L.LightningModule):
        self._log_step_as_current_epoch(trainer, L_module)

    def _log_step_as_current_epoch(self, trainer: L.Trainer, L_module: L.LightningModule):
        L_module.log("step", trainer.current_epoch)
