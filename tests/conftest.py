import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.torchsqlite import SqliteDataset


@pytest.fixture(scope="session")
def sample_dataframe():
    data = pd.DataFrame()
    data["time"] = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
    data["data1"] = [1, 2]
    data["data2"] = [10, 20]
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
