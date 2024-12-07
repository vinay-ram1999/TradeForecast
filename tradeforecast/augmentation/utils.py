from torch.utils.data import Dataset, Subset

import math

def train_val_test_split(dataset: Dataset, val_size: float=0.1, test_size: float=0.1) -> tuple[Subset]:
    dataset_len = len(dataset)
    test_start_idx = int(math.ceil((1.0 - test_size)*dataset_len))
    train_end_idx = int(math.ceil((1.0 - test_size - val_size)*dataset_len)) if val_size else test_start_idx
    train_dataset = Subset(dataset, range(0, train_end_idx))
    validation_dataset = Subset(dataset, range(train_end_idx, test_start_idx)) if val_size else None
    test_dataset = Subset(dataset, range(test_start_idx, dataset_len))
    if val_size:
        assert len(dataset) == len(train_dataset) + len(validation_dataset) + len(test_dataset), "Some data samples are missing"
        return (train_dataset, validation_dataset, test_dataset)
    else:
        assert len(dataset) == len(train_dataset) + len(test_dataset), "Some data samples are missing"
        return (train_dataset, test_dataset)
