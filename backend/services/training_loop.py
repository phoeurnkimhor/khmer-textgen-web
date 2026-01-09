import torch
import torch.nn as nn
import math
import os
from datetime import datetime

def training(model, train_loader, val_loader, vocab_size, stoi, itos, epochs=20, patience=3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)  # Move model to device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    wait = 0
    
    # Create temporary checkpoint directory
    temp_dir = "temp_checkpoints"
    os.makedirs(temp_dir, exist_ok=True)
    temp_checkpoint_path = f"{temp_dir}/best_model_temp.pt"

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_ppl = math.exp(avg_train_loss)

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits, _ = model(x_batch)
                loss = criterion(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_ppl = math.exp(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} (PPL {train_ppl:.2f}) | "
            f"Val Loss: {avg_val_loss:.4f} (PPL {val_ppl:.2f})"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0

            # Save temporary checkpoint (for recovery during training)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "vocab_size": vocab_size,
                "epoch": epoch + 1,
                "val_loss": avg_val_loss
            }
            torch.save(checkpoint, temp_checkpoint_path)

            print("  ** Validation improved, temporary checkpoint saved.")
        else:
            wait += 1
            print(f"  ** No improvement ({wait}/{patience})")

            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Load the best checkpoint back into the model
    if os.path.exists(temp_checkpoint_path):
        checkpoint = torch.load(temp_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from checkpoint (epoch {checkpoint['epoch']})")
    
    print(f"Training finished. Best Validation Loss: {best_val_loss:.4f}")
    
    return model, best_val_loss

