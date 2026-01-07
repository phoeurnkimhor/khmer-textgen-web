import re
import string

replace_map = {
    'ឝ': 'គ',
    'ឞ': 'ម',
}

unwanted_chars = [
    '\u200b', '\u200c', '\u200d', '\ufeff',
    '៙', '៚', '៖', 'ៗ', '៛', '៝', '៸', '៓'
]

khmer_punct = '។៕៘'

def clean_text(text: str) -> str:
    text = ''.join(c for c in text if c not in unwanted_chars)
    text = re.sub(r'[A-Za-z0-9]+', '', text)
    text = ''.join(c for c in text if c not in string.punctuation)
    text = re.sub(
        r'[^\u1780-\u17FF\u17E0-\u17E9\s' + khmer_punct + ']',
        '',
        text
    )
    for old, new in replace_map.items():
        text = text.replace(old, new)

    return re.sub(r'\s+', ' ', text).strip()
