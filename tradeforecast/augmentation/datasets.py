from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
from torch import Tensor
import polars as pl
import torch

from abc import abstractmethod
import math

class DatasetBase(Dataset):
    @abstractmethod
    def __read_lf__(self):
        pass

    @abstractmethod
    def __getitem__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __len__(self):
        pass

class RNNDataset(DatasetBase):
    def __init__(self, lf: pl.LazyFrame, train: bool, non_temporal: list, temporal: list, target: list, seq_length: int=30, split: float=0.20):
        self.lf = lf
        self.train_flag = train
        self.non_temporal = non_temporal
        self.temporal = temporal
        self.target = target
        self.seq_length = seq_length
        self.non_temporal_scaler = RobustScaler()
        self.tensors = self.__read_data__(split)
        pass

    def __read_data__(self, split) -> tuple[Tensor, ...]:
        lf_len = self.lf.select(pl.len()).collect().item()
        slice_idx = int(math.ceil((1.0 - split)*lf_len))
        self.lf = self.lf.slice(offset=0,length=slice_idx) if self.train_flag else self.lf.slice(offset=slice_idx,length=None)
        pl_df_non_temporal = self.lf.select(self.non_temporal).collect()
        self.non_temporal_scaler.set_output(transform='polars')
        pl_df_non_temporal = self.non_temporal_scaler.fit_transform(pl_df_non_temporal)
        #print(pl_df_non_temporal)
        X = self.lf.select(self.non_temporal).collect().to_numpy()
        y = self.lf.select(self.target).collect().to_numpy()
        return (torch.from_numpy(X).to(torch.float32), torch.from_numpy(y).to(torch.float32))
    
    def __getitem__(self, index):
        if index >= self.seq_length - 1:
            index_start = index - self.seq_length + 1
            x = self.X[index_start:(index + 1), :]
        else:
            padding = self.X[0].repeat(self.seq_length - index - 1, 1)
            x = self.X[0:(index + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[index]

    def __len__(self):
        return self.X.shape[0]
    