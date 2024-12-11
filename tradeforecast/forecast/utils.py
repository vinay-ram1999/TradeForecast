from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pytorch_lightning.callbacks import Callback
from torch import Tensor
import lightning as L
import numpy as np

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

def calc_metrics(y_actual: Tensor, y_predicted: Tensor):
    """Calculate performance metrics for models."""
    y_actual = y_actual.numpy()
    y_predicted = y_predicted.numpy()

    mae = mean_absolute_error(y_actual, y_predicted)
    mse = mean_squared_error(y_actual, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_predicted)

    return {"MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R-squared": r2}
