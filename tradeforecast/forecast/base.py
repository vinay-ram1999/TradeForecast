from torch.utils.data import DataLoader
from torch import nn, optim, Tensor
import torch

import os

from ..constants import models_dir

class BaseModel(nn.Module):
    @staticmethod
    def get_device_type() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __set_global_seed__(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def save_model_state(self, ticker_interval: str) -> str:
        model_fname = f'{ticker_interval}_{self}.pth'
        torch.save(self.state_dict(), os.path.join(models_dir, model_fname))
        return model_fname
    
    def load_model_state(self, model_fname: str):
        device = getattr(self, 'device')
        self.load_state_dict(torch.load(os.path.join(models_dir, model_fname), weights_only=True, map_location=device))
        print(f"Loaded '{model_fname.strip('.pth')}' model state_dict")
        self.eval()

class RNNBase(BaseModel):
    def train_model(self, 
                    criterion: nn.L1Loss | nn.MSELoss | nn.HuberLoss, 
                    optimizer: optim.Adam | optim.SGD | optim.Optimizer,
                    n_epochs: int,
                    data_loader: DataLoader,
                    min_learning_rate: float=0.0001):
        criterion = criterion()
        optimizer = optimizer(**{'params':self.parameters(),'lr':1.0})
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        device = getattr(self, 'device')
        self.train()
        for epoch in range(n_epochs):
            _epoch_loss = 0
            for train_x, train_y in data_loader:
                train_x: Tensor = train_x.to(device)
                train_y: Tensor = train_y.to(device)

                output: Tensor = self(train_x)
                loss: Tensor = criterion(output, train_y)
                _epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            _epoch_loss = _epoch_loss / len(data_loader)
            print(f'Epoch: {epoch+1}/{n_epochs}\tLR: {scheduler.get_last_lr()[-1]}\tLoss: {_epoch_loss:.6f}')
            if min_learning_rate < scheduler.get_last_lr()[-1]:
                scheduler.step()
        print("Training finished!!!")
        return
    
    @torch.inference_mode
    def test_model(self, data_loader: DataLoader) -> tuple[Tensor, Tensor]:
        assert torch.is_inference_mode_enabled(), "torch is not in inference_mode!"
        device = getattr(self, 'device')
        y_preds = []; y = []
        self.eval()
        for test_x, test_y in data_loader:
            test_x: Tensor = test_x.to(device)

            output: Tensor = self(test_x)
            y_preds += [output]
            y += [test_y]
        y = torch.cat(y, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        return (y, y_preds)
