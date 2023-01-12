import torch
from torch.utils.data import Dataset


class ASD_dataset(Dataset):
    def __init__(self, data):

        self.x = data

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])

        return x

    def __len__(self):
        return len(self.x)
