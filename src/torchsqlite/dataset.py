import sqlite3

import pandas as pd
from torch.utils.data import IterableDataset
import numpy as np


class SqliteDataset(IterableDataset):
    def __init__(self, filename: str, table_name: str, query: str, chunk_size: int = 10000) -> None:
        super().__init__()
        self.filename = filename
        self.table_name = table_name
        self.query = query
        self.chunk_size = chunk_size

        self.conn = sqlite3.connect(self.filename)
        self.conn.isolation_level = None  # Auto commit

    def insert(self, data: pd.DataFrame) -> None:
        if isinstance(data, pd.DataFrame):
            if data.isna().to_numpy().any():
                rows = np.where(data.isna().any(axis=1))[0].tolist()
                raise ValueError(f"Found NA/NAN values. Check df.iloc[{rows[0]}]")
        else:
            raise TypeError("data must be pd.DataFrame.")
        data.to_sql(name=self.table_name, con=self.conn, if_exists="append", index=False)

    def __iter__(self):
        cursor = self.conn.cursor()
        cursor.execute(self.query)

        while True:
            if not (chunk := cursor.fetchmany(self.chunk_size)):
                cursor.close()
                break
            yield from chunk

    def __len__(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count = cursor.fetchone()[0]
        cursor.close()
        return count
