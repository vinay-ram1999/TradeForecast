from torch.utils.data import Dataset
import polars as pl
import torch


class LSTMDataset(Dataset):
    def __init__(self, lf: pl.LazyFrame=None, target: list=['Close'], seq_length: int=32):
        self.seq_length = seq_length
        self.target = target
        self.features = lf.drop(target,).collect_schema().names()
        self.X = torch.tensor(lf.drop(target).collect().to_numpy())
        self.y = torch.tensor(lf.select(target,'Open').collect().to_numpy())
        print(self.X.ndim)
        pass

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        if index >= self.seq_length - 1:
            index_start = index - self.seq_length + 1
            x = self.X[index_start:(index + 1), :]
        else:
            padding = self.X[0].repeat(self.seq_length - index - 1, 1)
            x = self.X[0:(index + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[index]