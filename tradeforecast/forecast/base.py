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
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fname = f'{ticker_interval}_{self.__class__.__name__}_{n_params}.pth'
        torch.save(self.state_dict(), os.path.join(models_dir, model_fname))
        return model_fname
    
    def load_model_state(self, model_fname: str):
        device = getattr(self, 'device')
        self.load_state_dict(torch.load(os.path.join(models_dir, model_fname), weights_only=True, map_location=device))
        self.eval()

class RNNBase(BaseModel):    
    def train_model(self, 
                    criterion: nn.L1Loss | nn.MSELoss | nn.HuberLoss, 
                    optimizer: optim.Adam | optim.SGD | optim.Optimizer,
                    n_epochs: int,
                    data_loader: DataLoader,
                    learning_rate: float=0.001):
        criterion = criterion()
        optimizer = optimizer(**{'params':self.parameters(),'lr':learning_rate})
        device = getattr(self, 'device')
        for epoch in range(n_epochs):
            self.train()
            for train_x, train_y in data_loader:
                train_x: Tensor = train_x.to(device)

                output: Tensor = self(train_x)
                loss: Tensor = criterion(output, train_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss:.5f}')
        print("Training finished!!!")
        return
    
    @torch.inference_mode
    def test_model(self, data_loader: DataLoader) -> tuple[Tensor, Tensor]:
        assert torch.is_inference_mode_enabled(), "torch is not in inference_mode!"
        device = getattr(self, 'device')
        y_preds = []; y = []
        for test_x, test_y in data_loader:
            self.eval()
            test_x: Tensor = test_x.to(device)

            output: Tensor = self(test_x)
            y_preds += [output]
            y += [test_y]
        y = torch.cat(y, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        return (y, y_preds)
