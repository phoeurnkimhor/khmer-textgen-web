# backend/tests/services/test_text_generation_service.py
import torch
import types
import services.text_generation as tg


class DummyModel(torch.nn.Module):
    def forward(self, x):
        batch, seq = x.shape
        vocab_size = 3
        logits = torch.zeros(batch, seq, vocab_size)
        logits[:, :, 1] = 1.0  # always pick token index 1
        return logits, None


def test_generate_text_basic(monkeypatch):
    # Fake vocab
    fake_stoi = {"a": 0, "b": 1}
    fake_itos = {0: "a", 1: "b"}

    # Patch stoi / itos
    monkeypatch.setattr(tg, "stoi", fake_stoi)
    monkeypatch.setattr(tg, "itos", fake_itos)

    model = DummyModel()

    result = tg.generate_text(
        model=model,
        seed_text="ab",
        max_length=5,
        seq_len=2
    )

    assert isinstance(result, str)
    assert len(result) >= 2
    assert set(result).issubset({"a", "b"})


def test_generate_text_applies_cleaning(monkeypatch):
    fake_stoi = {"ស": 0}
    fake_itos = {0: "ស"}

    monkeypatch.setattr(tg, "stoi", fake_stoi)
    monkeypatch.setattr(tg, "itos", fake_itos)

    class StaticModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(1, x.shape[1], 1)
            return logits, None

    result = tg.generate_text(
        model=StaticModel(),
        seed_text="Hello123 ស",
        max_length=1,
        seq_len=1
    )

    assert result == "ស"
