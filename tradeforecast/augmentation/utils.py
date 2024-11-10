from torch.utils.data import Dataset, Subset

import math

def train_test_split(dataset: Dataset, split: float=0.2) -> tuple[Subset, Subset]:
    dataset_len = len(dataset)
    split_idx = int(math.ceil((1.0 - split)*dataset_len))
    train_dataset = Subset(dataset, range(0, split_idx))
    test_dataset = Subset(dataset, range(split_idx, dataset_len))
    return (train_dataset, test_dataset)
