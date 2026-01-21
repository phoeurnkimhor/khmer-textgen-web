import pandas as pd
from utils.build_vocab import build_vocab

def test_build_vocab_basic():
    df = pd.DataFrame({
        "sentence": ["abc", "bcd"]
    })

    all_text, vocab, stoi, itos = build_vocab(df)

    assert all_text == ["abc", "bcd"]
    assert set(vocab) == {"a", "b", "c", "d"}
    assert len(stoi) == len(itos)

def test_stoi_itos_inverse():
    df = pd.DataFrame({"sentence": ["ab"]})
    _, _, stoi, itos = build_vocab(df)

    for ch, idx in stoi.items():
        assert itos[idx] == ch
