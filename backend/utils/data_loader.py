from torch.utils.data import Dataset, DataLoader
import torch
from utils.split_data import split_data

class TSTDataset(Dataset):
    def __init__(self, texts, seq_len=50, stoi=None):
        self.seq_len = seq_len
        self.data = []
        for sentence in texts:
            token_ids = [stoi[ch] for ch in list(sentence) if ch in stoi]
            for i in range(len(token_ids) - seq_len):
                self.data.append((token_ids[i:i+seq_len], token_ids[i+1:i+seq_len+1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

def get_dataloader(texts, seq_len=50, batch_size=32, stoi=None):

    train_texts, val_texts, test_texts = split_data(texts)

    # Create datasets
    train_dataset = TSTDataset(train_texts, seq_len=seq_len, stoi=stoi)
    val_dataset = TSTDataset(val_texts, seq_len=seq_len, stoi=stoi)
    test_dataset = TSTDataset(test_texts, seq_len=seq_len, stoi=stoi)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader