from utils.preprocessing import split_sentences, chunk_text

def test_split_sentences_basic():
    text = "ខ្ញុំសុខសប្បាយ។អ្នកយ៉ាងដូចម្តេច៕"
    result = split_sentences(text)
    assert result == ["ខ្ញុំសុខសប្បាយ", "អ្នកយ៉ាងដូចម្តេច"]

def test_split_sentences_removes_empty():
    text = "។៕"
    result = split_sentences(text)
    assert result == []

def test_chunk_text_basic():
    sentence = "abcdefghij"
    chunks = chunk_text(sentence, chunk_size=4)
    assert chunks == ["abcd", "efgh", "ij"]
