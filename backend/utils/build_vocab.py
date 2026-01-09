def build_vocab(df):
    all_text = df['sentence'].tolist()
    tokens = [t for sentence in all_text for t in list(sentence)]
    vocab = sorted(set(tokens))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return all_text, vocab, stoi, itos
