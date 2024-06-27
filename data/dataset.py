import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

    def batch(self, batch_size):
        for i in range(0, len(self.texts), batch_size):
            yield self.texts[i:i + batch_size]
