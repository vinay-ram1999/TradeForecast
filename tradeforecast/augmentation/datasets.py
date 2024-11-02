from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import RobustScaler
from sklearn.exceptions import NotFittedError
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
        self.features = self.non_temporal + self.temporal
        self.target = target
        self.seq_length = seq_length
        self.non_temporal_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.tensors = self.__read_data__(split)
        pass

    @property
    def __lf_len__(self) -> int:
        return self.lf.select(pl.len()).collect().item()

    def __read_data__(self, split) -> tuple[Tensor, ...]:
        slice_idx = int(math.ceil((1.0 - split)*self.__lf_len__))
        self.lf = self.lf.slice(offset=0,length=slice_idx) if self.train_flag else self.lf.slice(offset=slice_idx,length=None)
        temporal_tensor: Tensor = self.lf.select(self.temporal).slice(0, self.__lf_len__ - 1).collect().to_torch(return_type='tensor', dtype=pl.Float32)
        non_temporal_pl_df = self.lf.select(self.non_temporal).slice(0, self.__lf_len__ - 1).collect()
        target_pl_df = self.lf.with_columns(pl.col(self.target).shift(n=-1)).select(self.target).drop_nulls().collect()
        target_tensor: Tensor = self.fit_transform(target_pl_df, self.target_scaler)
        non_temporal_tensor: Tensor = self.fit_transform(non_temporal_pl_df, self.non_temporal_scaler)
        features_tensor = torch.cat((non_temporal_tensor, temporal_tensor), dim=1)
        X = torch.empty((features_tensor.size(0) - self.seq_length + 1, self.seq_length, features_tensor.size(1)), dtype=torch.float32)
        y = torch.empty((target_tensor.size(0) - self.seq_length + 1, self.seq_length, target_tensor.size(1)), dtype=torch.float32)
        for i in range(X.size(0)):
            for j in range(self.seq_length):
                X[i][j][:] = features_tensor[i+j][:]
                y[i][j][:] = target_tensor[i+j][:]
        return (X, y)
    
    def fit_transform(self, pl_df: pl.DataFrame, scaler: RobustScaler) -> Tensor:
        try:
            check_is_fitted(scaler)
            raise ValueError(f"'{scaler}' is already fitted!")
        except NotFittedError:
            scaler.set_output(transform='polars')
            pl_df: pl.DataFrame = scaler.fit_transform(pl_df)
            return pl_df.to_torch(return_type='tensor', dtype=pl.Float32)
    
    def inverse_transform(self) -> Tensor:
        return
    
    def __len__(self):
        return self.tensors[0].size(0)
    
    def __getitem__(self, idx):
        return tuple(tensor[idx] for tensor in self.tensors)
