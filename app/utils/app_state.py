"""
AppState — singleton that shares the active dataset across all pages.

Supports two modes:
  • Normal (pandas):  df is a full DataFrame in memory
  • Large (DuckDB):   df holds only a preview; full data is queried via `duckdb_con`
"""

import pandas as pd
from typing import Optional


class AppState:
    _dataframe:   Optional[pd.DataFrame] = None
    _filepath:    Optional[str]          = None
    _is_large:    bool                   = False   # True → DuckDB mode
    _duckdb_con                          = None    # duckdb.DuckDBPyConnection | None
    _row_count:   int                    = 0
    _col_count:   int                    = 0
    _column_names: list                  = []
    _column_types: dict                  = {}      # {col_name: dtype_str}

    # ------------------------------------------------------------------ #
    #  Setters                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def set_dataframe(cls, df: pd.DataFrame, filepath: str = ""):
        """Standard in-memory mode."""
        cls._dataframe    = df
        cls._filepath     = filepath
        cls._is_large     = False
        cls._duckdb_con   = None
        cls._row_count    = len(df)
        cls._col_count    = len(df.columns)
        cls._column_names = list(df.columns)
        cls._column_types = {c: str(df[c].dtype) for c in df.columns}

    @classmethod
    def set_duckdb(cls, con, filepath: str = ""):
        """Large-file DuckDB mode — called by DataLoader, not UI code."""
        cls._duckdb_con = con
        cls._filepath   = filepath
        cls._is_large   = True
        # dataframe holds only the preview rows; set by DataLoader separately
        # via set_preview_df below

    @classmethod
    def set_preview_df(cls, df: pd.DataFrame):
        cls._dataframe    = df
        cls._row_count    = 0        # will be overridden by set_meta
        cls._col_count    = len(df.columns)
        cls._column_names = list(df.columns)

    @classmethod
    def set_meta(cls, row_count: int, col_count: int,
                 column_names: list, column_types: dict):
        cls._row_count    = row_count
        cls._col_count    = col_count
        cls._column_names = column_names
        cls._column_types = column_types

    # ------------------------------------------------------------------ #
    #  Getters                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_dataframe(cls) -> Optional[pd.DataFrame]:
        """Returns the full DataFrame (normal mode) or preview rows (large mode)."""
        return cls._dataframe

    @classmethod
    def get_duckdb(cls):
        """Returns the DuckDB connection, or None in normal mode."""
        return cls._duckdb_con

    @classmethod
    def is_large(cls) -> bool:
        return cls._is_large

    @classmethod
    def get_filepath(cls) -> Optional[str]:
        return cls._filepath

    @classmethod
    def get_row_count(cls) -> int:
        return cls._row_count

    @classmethod
    def get_col_count(cls) -> int:
        return cls._col_count

    @classmethod
    def get_column_names(cls) -> list:
        return cls._column_names

    @classmethod
    def get_column_types(cls) -> dict:
        return cls._column_types

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def has_data(cls) -> bool:
        return cls._dataframe is not None or cls._duckdb_con is not None

    @classmethod
    def clear(cls):
        if cls._duckdb_con is not None:
            try:
                cls._duckdb_con.close()
            except Exception:
                pass
        cls._dataframe    = None
        cls._filepath     = None
        cls._is_large     = False
        cls._duckdb_con   = None
        cls._row_count    = 0
        cls._col_count    = 0
        cls._column_names = []
        cls._column_types = {}
