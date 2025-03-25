def test_dataset_len(sample_dataset, sample_dataframe):
    assert len(sample_dataset) == len(sample_dataframe)


def test_dataset_lter(sample_dataset):
    for row in sample_dataset:
        assert len(row) > 0


def test_rolling_dataset_lter(sample_rolling_dataset):
    for row in sample_rolling_dataset:
        assert row.shape[0] == sample_rolling_dataset.window
