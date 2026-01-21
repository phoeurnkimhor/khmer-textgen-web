import pytest
import torch


@pytest.fixture
def fake_vocab():
    stoi = {"a": 0, "b": 1}
    itos = {0: "a", 1: "b"}
    return stoi, itos


@pytest.fixture
def dummy_model():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            batch, seq = x.shape
            vocab_size = 2
            logits = torch.zeros(batch, seq, vocab_size)
            logits[:, :, 1] = 1.0  # always predict "b"
            return logits, None

    return DummyModel()


@pytest.fixture
def sample_seed_text():
    return "ab"
