import pickle
import torch
from torch.utils.data import Dataset

class CustomizedDataset(Dataset):
    def __init__(self, metadata):
        super(CustomizedDataset, self).__init__()
        self.data = pickle.load(open(metadata, 'rb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with torch.no_grad():
            text_embed = self.data[idx]['text_embed']
            audio_embed = self.data[idx]['audio_embed']
            label = self.data[idx]['label']
        return text_embed, audio_embed, label
