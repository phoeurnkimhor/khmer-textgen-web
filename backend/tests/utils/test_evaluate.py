import torch
from utils.evaluate import evaluate_model

class DummyModel(torch.nn.Module):
    def forward(self, x):
        batch, seq = x.shape
        vocab_size = 5
        logits = torch.zeros(batch, seq, vocab_size)
        return logits, None

def test_evaluate_model_outputs():
    model = DummyModel()
    x = torch.zeros((2, 3), dtype=torch.long)
    y = torch.zeros((2, 3), dtype=torch.long)

    dataloader = [(x, y)]

    avg_loss, perplexity, accuracy = evaluate_model(model, dataloader)

    assert isinstance(avg_loss, float)
    assert isinstance(perplexity, float)
    assert 0.0 <= accuracy <= 1.0
