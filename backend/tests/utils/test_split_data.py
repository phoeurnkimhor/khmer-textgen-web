from utils.split_data import split_data

def test_split_data_ratios():
    data = list(range(10))
    train, val, test = split_data(data)

    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1

def test_split_data_preserves_order():
    data = ["a", "b", "c", "d"]
    train, val, test = split_data(data, 0.5, 0.25, 0.25)

    assert train == ["a", "b"]
    assert val == ["c"]
    assert test == ["d"]
