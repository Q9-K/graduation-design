from torch.utils.data import Dataset
import os

class MyDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
