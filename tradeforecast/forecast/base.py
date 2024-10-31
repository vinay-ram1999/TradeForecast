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
                    x: Tensor, 
                    criterion: nn.L1Loss | nn.MSELoss | nn.HuberLoss, 
                    optimizer: optim.Adam | optim.SGD | optim.Optimizer,
                    n_epochs: int,
                    data_loader: DataLoader,
                    learning_rate: float=0.001):
        self.train()
        criterion = criterion()
        optimizer = optimizer(**{'params':self.parameters(),'lr':learning_rate})
        hidden_size = getattr(self, 'hidden_size')
        input_size = getattr(self, 'input_size')
        for epoch in range(n_epochs):
            self.train()
            for train_x, train_y in data_loader:
                train_x: Tensor; train_y: Tensor
                train_x = train_x.view(-1, hidden_size, input_size)
                
                output = self(train_x)
                loss: Tensor = criterion(output, train_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss:.5f}')
        print("Training finished!!!")
        return
    
    @torch.inference_mode
    def predict(self):
        assert torch.is_inference_mode_enabled(), "torch is not in inference_mode!"
        print(self.fc_linear)
        self.eval()
        return
