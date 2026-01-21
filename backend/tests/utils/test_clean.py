from utils.clean import clean_text

def test_remove_english_and_numbers():
    text = "Hello 123 សួស្តី"
    cleaned = clean_text(text)
    assert "Hello" not in cleaned
    assert "123" not in cleaned
    assert "សួស្តី" in cleaned

def test_replace_map():
    text = "ឝឞ"
    cleaned = clean_text(text)
    assert cleaned == "គម"

def test_whitespace_normalization():
    text = "សួស្តី     លោក"
    cleaned = clean_text(text)
    assert cleaned == "សួស្តី លោក"
