from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import RobustScaler
from sklearn.exceptions import NotFittedError
from torch.utils.data import Dataset
from torch import Tensor
import polars as pl
import numpy as np
import torch

from abc import abstractmethod

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
    def __init__(self, 
                 lf: pl.LazyFrame,
                 non_temporal: list, temporal: list, target: str, 
                 look_back_len: int=30, forecast_len: int=5):    # by default we look_back 30 samples to forecast 5 new samples
        assert forecast_len < look_back_len, "forecast_len >= look_back_len is not considered as optimal"
        assert isinstance(target, str), "target variable must be a string"
        self.lf = lf
        self.non_temporal = non_temporal
        self.temporal = temporal
        self.features = self.non_temporal + self.temporal
        self.target = target
        self.look_back_len = look_back_len
        self.forecast_len = forecast_len
        self.non_temporal_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.tensors = self.__read_data__()
        assert all(self.tensors[0].size(0) == tensor.size(0) for tensor in self.tensors), "Size mismatch between tensors"

    @property
    def __lf_len__(self) -> int:
        return self.lf.select(pl.len()).collect().item()

    def __read_data__(self) -> tuple[Tensor, ...]:
        non_temporal_pl_df = self.lf.select(self.non_temporal).slice(0, self.__lf_len__ - self.look_back_len).collect()
        target_pl_df = self.lf.with_columns(pl.col(self.target).shift(n=-self.look_back_len).alias('shifted')).select('shifted').drop_nulls().collect()
        target_tensor: Tensor = self.fit_transform(target_pl_df, self.target_scaler)
        non_temporal_tensor: Tensor = self.fit_transform(non_temporal_pl_df, self.non_temporal_scaler)
        if self.temporal == list():
            features_tensor = non_temporal_tensor
        else:
            temporal_tensor: Tensor = self.lf.select(self.temporal).slice(0, self.__lf_len__ - self.look_back_len).collect().to_torch(return_type='tensor', dtype=pl.Float32)
            features_tensor = torch.cat((non_temporal_tensor, temporal_tensor), dim=1)
        X = []; y = []
        for i in range(features_tensor.size(0) - self.look_back_len - self.forecast_len):
            look_back_idx = i + self.look_back_len
            forecast_idx = look_back_idx + self.forecast_len
            X += [features_tensor[i:look_back_idx]]
            y += [target_tensor[look_back_idx:forecast_idx]]
        X = torch.from_numpy(np.array(X).reshape(-1, self.look_back_len, features_tensor.size(1))).to(torch.float32)
        y = torch.from_numpy(np.array(y).reshape(-1, self.forecast_len)).to(torch.float32)
        return (X, y)
    
    def fit_transform(self, pl_df: pl.DataFrame, scaler: RobustScaler) -> Tensor:
        scaler.set_output(transform='polars')
        pl_df: pl.DataFrame = scaler.fit_transform(pl_df)
        return pl_df.to_torch(return_type='tensor', dtype=pl.Float32)
    
    def inverse_transform(self, y: Tensor) -> pl.DataFrame:
        try:
            check_is_fitted(self.target_scaler)
            y: pl.DataFrame = self.target_scaler.inverse_transform(y)
            return y
        except NotFittedError as e:
            print(e.add_note(f"'{self.target_scaler}' is not fitted yet"))
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return tuple(tensor[idx] for tensor in self.tensors)
    
    def __len__(self):
        return self.tensors[0].size(0)
