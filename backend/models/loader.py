import torch
from .architecture import LSTMTST

vocab = [
    ' ', 'ក', 'ខ', 'គ', 'ឃ', 'ង', 'ច', 'ឆ', 'ជ', 'ឈ', 'ញ', 'ដ', 'ឋ', 'ឌ', 'ឍ', 'ណ',
    'ត', 'ថ', 'ទ', 'ធ', 'ន', 'ប', 'ផ', 'ព', 'ភ', 'ម', 'យ', 'រ', 'ល', 'វ', 'ឝ', 'ឞ',
    'ស', 'ហ', 'ឡ', 'អ', 'ឣ', 'ឤ', 'ឥ', 'ឦ', 'ឧ', 'ឩ', 'ឪ', 'ឫ', 'ឬ', 'ឭ', 'ឮ',
    'ឯ', 'ឰ', 'ឱ', 'ឲ', 'ឳ', 'ា', 'ិ', 'ី', 'ឹ', 'ឺ', 'ុ', 'ូ', 'ួ', 'ើ', 'ឿ',
    'ៀ', 'េ', 'ែ', 'ៃ', 'ោ', 'ៅ', 'ំ', 'ះ', 'ៈ', '៉', '៊', '់', '៌', '៍', '៏',
    '័', '៑', '្', '៝', '០', '១', '២', '៣', '៤', '៥', '៦', '៧', '៨', '៩'
]

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}

def load_model(checkpoint_path: str):
    model = LSTMTST(vocab_size=len(stoi))
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
