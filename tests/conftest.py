import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.torchsqlite import SqliteDataset, RollingSqliteDataset


@pytest.fixture(scope="session")
def sample_dataframe():
    data = pd.DataFrame()
    data["time"] = [
        datetime(2025, 1, 1),
        datetime(2025, 1, 2),
        datetime(2025, 1, 3),
        datetime(2025, 1, 4),
        datetime(2025, 1, 5),
    ]
    data["data1"] = [1, 2, 3, 4, 5]
    data["data2"] = [10, 20, 30, 40, 50]
    yield data


@pytest.fixture(scope="session")
def sample_dataset(sample_dataframe):
    filename = "sample_dataset.db"
    table_name = "data"
    query = f"SELECT * FROM {table_name}"
    dataset = SqliteDataset(filename=filename, table_name=table_name, query=query)
    dataset.insert(sample_dataframe)
    yield dataset
    dataset.conn.close()
    Path(filename).unlink()


@pytest.fixture(scope="session")
def sample_rolling_dataset(sample_dataframe):
    filename = "sample_rolling_dataset.db"
    table_name = "data"
    query = f"SELECT data1, data2 FROM {table_name}"
    window = 3
    dataset = RollingSqliteDataset(filename=filename, table_name=table_name, query=query, window=window)
    dataset.insert(sample_dataframe)
    yield dataset
    dataset.conn.close()
    Path(filename).unlink()
