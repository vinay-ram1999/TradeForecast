from torch import nn
import torch

from ..constants import models_dir

class BaseModel(nn.Module):
    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def save_model_state(self):
        return