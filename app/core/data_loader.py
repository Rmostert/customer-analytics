"""
DataLoader — reads CSV, Excel, JSON, and Parquet files into a pandas DataFrame.

For CSV and Parquet files that exceed the memory threshold, the file is NOT loaded
into memory. A DuckDB connection is returned inside LoadResult so the UI thread can
safely store it in AppState after the worker finishes.

IMPORTANT: DataLoader never touches AppState — it is called from a background
thread, and AppState interactions must happen on the main (Qt) thread only.
"""

import os
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any


# Files larger than this are handled via DuckDB instead of pandas.
MEMORY_THRESHOLD_BYTES = 500 * 1024 * 1024   # 500 MB

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


@dataclass
class LoadResult:
    """
    Returned by DataLoader.load().
    - Normal files:  df contains the full DataFrame, duckdb_con is None.
    - Large CSV / Parquet: df contains only the first 100 preview rows,
                           duckdb_con holds the open connection for later queries.
    Always check is_large before using df as the full dataset.
    """
    df:           Optional[pd.DataFrame]
    filepath:     str
    extension:    str
    is_large:     bool = False
    row_count:    int  = 0
    col_count:    int  = 0
    column_names: list = field(default_factory=list)
    column_types: dict = field(default_factory=dict)
    duckdb_con:   Any  = None   # duckdb.DuckDBPyConnection | None
    load_encoding: str = "utf-8"  # text encoding used for CSV (large or small)


class DataLoader:

    @staticmethod
    def _escape_path(filepath: str) -> str:
        return filepath.replace("'", "''")

    @staticmethod
    def _duckdb_csv_encoding(encoding: str) -> str:
        mapping = {
            "utf-8":  "utf-8",
            "utf-16": "utf-16",
            "latin-1": "latin-1",
            "cp1252": "latin-1",  # DuckDB has no cp1252; closest single-byte fallback
        }
        return mapping.get(encoding.lower(), "utf-8")

    @staticmethod
    def _dataset_view_sql(filepath: str, extension: str, encoding: str = "utf-8") -> str:
        safe = DataLoader._escape_path(filepath)
        if extension == ".parquet":
            return f"SELECT * FROM read_parquet('{safe}')"
        if extension == ".csv":
            enc = DataLoader._duckdb_csv_encoding(encoding)
            return (
                f"SELECT * FROM read_csv_auto('{safe}', "
                f"header=true, encoding='{enc}')"
            )
        raise ValueError(f"DuckDB out-of-core loading is not supported for '{extension}'.")

    @staticmethod
    def _load_duckdb(filepath: str, extension: str, encoding: str = "utf-8") -> LoadResult:
        """
        Large file path — opens a DuckDB connection and queries metadata only.
        The full dataset is never loaded into RAM; all later queries run via SQL.
        """
        import duckdb

        con = duckdb.connect(database=":memory:")
        view_sql = DataLoader._dataset_view_sql(filepath, extension, encoding)
        con.execute(f"CREATE VIEW dataset AS {view_sql}")

        schema_df    = con.execute("DESCRIBE dataset").df()
        column_names = schema_df["column_name"].tolist()
        column_types = dict(zip(schema_df["column_name"], schema_df["column_type"]))
        row_count    = con.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
        preview_df   = con.execute("SELECT * FROM dataset LIMIT 100").df()

        return LoadResult(
            df=preview_df,
            filepath=filepath,
            extension=extension,
            is_large=True,
            row_count=row_count,
            col_count=len(column_names),
            column_names=column_names,
            column_types=column_types,
            duckdb_con=con,
            load_encoding=encoding,
        )

    @staticmethod
    def load(filepath: str, encoding: str = "utf-8") -> LoadResult:
        path = Path(filepath)
        ext  = path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if ext == ".parquet":
            return DataLoader._load_parquet(filepath)
        elif ext == ".csv":
            return DataLoader._load_csv(filepath, encoding)
        elif ext in {".xlsx", ".xls"}:
            return DataLoader._load_excel(filepath)
        elif ext == ".json":
            return DataLoader._load_json(filepath, encoding)

    # ------------------------------------------------------------------ #
    #  Parquet                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_parquet(filepath: str) -> LoadResult:
        file_size = os.path.getsize(filepath)

        if file_size > MEMORY_THRESHOLD_BYTES:
            return DataLoader._load_duckdb(filepath, ".parquet")
        df = pd.read_parquet(filepath)
        return DataLoader._wrap(df, filepath, ".parquet")

    # ------------------------------------------------------------------ #
    #  CSV / Excel / JSON                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_csv(filepath: str, encoding: str) -> LoadResult:
        file_size = os.path.getsize(filepath)

        if file_size > MEMORY_THRESHOLD_BYTES:
            try:
                return DataLoader._load_duckdb(filepath, ".csv", encoding)
            except Exception:
                if encoding != "latin-1":
                    return DataLoader._load_duckdb(filepath, ".csv", "latin-1")
                raise

        try:
            df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            encoding = "latin-1"
            df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
        return DataLoader._wrap(df, filepath, ".csv", encoding=encoding)

    @staticmethod
    def _load_excel(filepath: str) -> LoadResult:
        df  = pd.read_excel(filepath, engine="openpyxl")
        ext = Path(filepath).suffix.lower()
        return DataLoader._wrap(df, filepath, ext)

    @staticmethod
    def _load_json(filepath: str, encoding: str) -> LoadResult:
        try:
            df = pd.read_json(filepath, encoding=encoding)
        except ValueError:
            df = pd.read_json(filepath, orient="records", encoding=encoding)
        return DataLoader._wrap(df, filepath, ".json")

    @staticmethod
    def _wrap(df: pd.DataFrame, filepath: str, ext: str,
              encoding: str = "utf-8") -> LoadResult:

        return LoadResult(
            df=df,
            filepath=filepath,
            extension=ext,
            is_large=False,
            row_count=len(df),
            col_count=len(df.columns),
            column_names=list(df.columns),
            column_types={c: str(df[c].dtype) for c in df.columns},
            duckdb_con=None,
            load_encoding=encoding,
        )
