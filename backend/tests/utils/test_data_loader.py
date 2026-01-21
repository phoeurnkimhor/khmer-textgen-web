import torch
from utils.data_loader import TSTDataset, get_dataloader

def test_tst_dataset_length():
    texts = ["abcd"]
    stoi = {"a": 0, "b": 1, "c": 2, "d": 3}

    dataset = TSTDataset(texts, seq_len=2, stoi=stoi)
    assert len(dataset) == 2  # ab→bc, bc→cd

def test_tst_dataset_item_shape():
    texts = ["abcd"]
    stoi = {"a": 0, "b": 1, "c": 2, "d": 3}

    dataset = TSTDataset(texts, seq_len=2, stoi=stoi)
    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 2
    assert y.shape[0] == 2

def test_get_dataloader_returns_loaders():
    texts = ["abcdef"]
    stoi = {ch: i for i, ch in enumerate("abcdef")}

    train_loader, val_loader, test_loader = get_dataloader(
        texts, seq_len=2, batch_size=2, stoi=stoi
    )

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None
