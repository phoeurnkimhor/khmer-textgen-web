import math
import torch

def evaluate_model(model, dataloader, device="cpu"):
    model.eval()  
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():  # no gradients needed for evaluation
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits, _ = model(x_batch)

            # compute token-level accuracy
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y_batch).sum().item()
            total_tokens += y_batch.numel()

    avg_loss = total_loss / len(dataloader.dataset)
    perplexity = math.exp(avg_loss)
    accuracy = total_correct / total_tokens

    return avg_loss, perplexity, accuracy