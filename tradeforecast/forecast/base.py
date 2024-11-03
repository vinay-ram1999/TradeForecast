from torch.utils.data import DataLoader
from torch import nn, optim, Tensor
import torch

from ..constants import models_dir

class BaseModel(nn.Module):
    @staticmethod
    def get_device_type():
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def save_model_state(self):
        return

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
        output_size = getattr(self, 'output_size')
        for epoch in range(n_epochs):
            self.train()
            for train_x, train_y in data_loader:
                train_x: Tensor = train_x.to(device)
                train_y: Tensor = train_y.view(-1, output_size).to(device)

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
        output_size = getattr(self, 'output_size')
        y_preds = []; y = []
        for test_x, test_y in data_loader:
            self.eval()
            test_x: Tensor = test_x.to(device)
            test_y: Tensor = test_y.view(-1, output_size)

            output: Tensor = self(test_x)
            y_preds += [output]
            y += [test_y]
        y = torch.cat(y, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        return (y, y_preds)
