from src.dataset import split_train_test_validate

def test_split_train_test_validate():
    data = ['a','ab','cd','ed','as','aw','ap','223','aa','aad','ds','RQ','wsdqw','wdew']
    result = split_train_test_validate(data, (1,3,2))
    assert isinstance(result, dict) ,"Result must be a dict"
    assert len(result["train"]) == 2
    assert len(result["test"]) == 7
    assert len(result["val"]) == 5
    assert all(map(lambda e: e in data, list(result["train"])))

    result2 = split_train_test_validate(data, (1,3,2))
    for split in result2:
        assert result2[split] == result[split], "Result must be consistent"

