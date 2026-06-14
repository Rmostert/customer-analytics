"""
DuckDB helpers for out-of-core segmentation on large CSV / Parquet files.

Opens its own in-memory connections (safe to call from worker threads).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterator, Optional

import duckdb
import pandas as pd

from app.core.data_loader import DataLoader


DEFAULT_SAMPLE_SIZE = 50_000
MAX_SAMPLE_SIZE = 500_000
DEFAULT_BATCH_SIZE = 50_000


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def open_dataset_view(filepath: str, encoding: str = "utf-8"):
    """Return a DuckDB connection with a `dataset` view over the file."""
    ext = Path(filepath).suffix.lower()
    con = duckdb.connect(database=":memory:")
    view_sql = DataLoader._dataset_view_sql(filepath, ext, encoding)
    con.execute(f"CREATE VIEW dataset AS {view_sql}")
    return con


def fetch_sample(
    filepath: str,
    columns: list[str],
    n_rows: int,
    seed: int = 42,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Reservoir sample of *n_rows* from the file."""
    con = open_dataset_view(filepath, encoding)
    try:
        cols_sql = ", ".join(quote_identifier(c) for c in columns)
        n = max(1, min(int(n_rows), MAX_SAMPLE_SIZE))
        sql = f"SELECT {cols_sql} FROM dataset USING SAMPLE {n} ROWS (reservoir, {seed})"
        return con.execute(sql).df()
    finally:
        con.close()


def fetch_quantile_bins(filepath: str, column: str, encoding: str = "utf-8") -> list[float]:
    """Return [min, Q1, Q2, Q3, max] cut-points for *column* over the full file."""
    con = open_dataset_view(filepath, encoding)
    col = quote_identifier(column)
    try:
        row = con.execute(f"""
            SELECT
                quantile_cont({col}, 0.0)  AS q0,
                quantile_cont({col}, 0.25) AS q1,
                quantile_cont({col}, 0.50) AS q2,
                quantile_cont({col}, 0.75) AS q3,
                quantile_cont({col}, 1.0)  AS q4
            FROM dataset
            WHERE {col} IS NOT NULL
        """).fetchone()
        if row is None:
            raise ValueError(f"Column '{column}' has no non-null values.")
        qs = list(row)
        result = [qs[0]]
        for v in qs[1:]:
            if v > result[-1]:
                result.append(v)
        return result
    finally:
        con.close()


def iter_buckets(
    filepath: str,
    columns: list[str],
    id_col: str,
    total_rows: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    encoding: str = "utf-8",
) -> Iterator[pd.DataFrame]:
    """
    Yield successive row batches without loading the full file.
    Uses hash bucketing on *id_col* for roughly even splits.
    """
    n_buckets = max(1, math.ceil(total_rows / batch_size))
    cols_sql = ", ".join(quote_identifier(c) for c in columns)
    id_sql = quote_identifier(id_col)
    con = open_dataset_view(filepath, encoding)
    try:
        for bucket in range(n_buckets):
            sql = f"""
                SELECT {cols_sql}
                FROM dataset
                WHERE (hash({id_sql}) % {n_buckets}) = {bucket}
            """
            batch = con.execute(sql).df()
            if not batch.empty:
                yield batch
    finally:
        con.close()


def score_in_batches(
    filepath: str,
    columns: list[str],
    id_col: str,
    total_rows: int,
    score_batch: Callable[[pd.DataFrame], pd.DataFrame],
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: Optional[Callable[[int], None]] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Apply *score_batch* to each bucket and concatenate labelled rows."""
    parts: list[pd.DataFrame] = []
    n_buckets = max(1, math.ceil(total_rows / batch_size))
    for i, batch in enumerate(
        iter_buckets(filepath, columns, id_col, total_rows, batch_size, encoding)
    ):
        scored = score_batch(batch)
        if not scored.empty:
            parts.append(scored)
        if progress:
            progress(int(20 + 60 * (i + 1) / n_buckets))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
