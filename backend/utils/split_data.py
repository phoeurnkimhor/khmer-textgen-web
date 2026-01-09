def split_data(all_text, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):    

    n_total = len(all_text)
    train_end = int(n_total * train_ratio)
    val_end = train_end + int(n_total * val_ratio)

    train_texts = all_text[:train_end]
    val_texts = all_text[train_end:val_end]
    test_texts = all_text[val_end:]

    return train_texts, val_texts, test_texts
