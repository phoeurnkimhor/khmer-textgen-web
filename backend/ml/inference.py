import torch
from torch.nn.utils.rnn import pad_sequence

from .architecture import LSTMAutoencoder
from utils.preprocessing import clean_text

DEVICE = "cpu"

class AutoencoderInference:
    def __init__(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        self.stoi = checkpoint["stoi"]
        self.itos = {int(k): v for k, v in checkpoint["itos"].items()}
        self.pad_id = self.stoi["<pad>"]

        self.model = LSTMAutoencoder(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"]
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
        ids = (
            [self.stoi["<sos>"]]
            + [self.stoi.get(c, self.stoi["<unk>"]) for c in text]
            + [self.stoi["<eos>"]]
        )
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        tokens = [
            self.itos[i]
            for i in ids
            if self.itos[i] not in ("<sos>", "<eos>", "<pad>", "<unk>")
        ]
        return "".join(tokens)

    @torch.no_grad()
    def predict(self, raw_text: str) -> str:
        text = clean_text(raw_text)

        x = self.encode(text)
        x = pad_sequence([x], batch_first=True, padding_value=self.pad_id)

        logits = self.model(x)
        pred_ids = logits.argmax(dim=-1)[0].tolist()

        return self.decode(pred_ids)
    
model = AutoencoderInference(checkpoint_path="./best_autoencoder_lstm.pt")
