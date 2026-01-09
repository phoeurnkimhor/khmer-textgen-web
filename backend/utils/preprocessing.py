import re

def split_sentences(text):
    sentences = re.split(r'[។៕]', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(sentence, chunk_size=120):
    return [sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size)]
