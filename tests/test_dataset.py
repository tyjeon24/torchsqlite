def test_dataset_len(sample_dataset, sample_dataframe):
    assert len(sample_dataset) == len(sample_dataframe)


def test_dataset_lter(sample_dataset):
    for row in sample_dataset:
        assert len(row) > 0
