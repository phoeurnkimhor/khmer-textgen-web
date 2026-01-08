import torch
from models.loader import stoi, itos
from utils.preprocessing import clean_text 

def generate_text(model, seed_text, max_length=100, seq_len=50):
    
    seed_text = clean_text(seed_text)

    tokens = [stoi[ch] for ch in seed_text if ch in stoi]
    generated = tokens.copy()

    for _ in range(max_length):
        input_seq = torch.tensor([generated[-seq_len:]])
        with torch.no_grad():
            logits, _ = model(input_seq)

        next_token = logits[:, -1, :].argmax(dim=-1).item()
        generated.append(next_token)

    return "".join(itos[t] for t in generated)
