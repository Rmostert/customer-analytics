"""
DataProfiler — generates per-column statistics for the Explore page.

Two code paths:
  • pandas path  — dataset fits in memory (CSV, Excel, JSON, small Parquet)
  • DuckDB path  — large Parquet file; all aggregations run as SQL queries
                   so only the results come back into memory, never the raw data.
"""

import pandas as pd
import numpy as np
from app.utils.app_state import AppState


class DataProfiler:

    @staticmethod
    def profile(df: pd.DataFrame) -> dict:
        """
        Main entry point.  Delegates to DuckDB path automatically when
        AppState.is_large() is True.
        """
        if AppState.is_large():
            con = AppState.get_duckdb()
            if con is None:
                raise RuntimeError("DuckDB connection not available.")
            return DataProfiler._profile_duckdb(con)
        else:
            return DataProfiler._profile_pandas(df)

    # ------------------------------------------------------------------ #
    #  Pandas path                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _profile_pandas(df: pd.DataFrame) -> dict:
        columns = {}

        for col in df.columns:
            series     = df[col]
            dtype      = str(series.dtype)
            is_numeric = pd.api.types.is_numeric_dtype(series)

            count       = int(series.count())
            unique      = int(series.nunique())
            missing     = int(series.isnull().sum())
            missing_pct = round(missing / len(df) * 100, 1) if len(df) > 0 else 0

            stats = {
                "dtype":          dtype,
                "dtype_category": "numeric" if is_numeric else "categorical",
                "count":          count,
                "unique":         unique,
                "missing":        missing,
                "missing_pct":    missing_pct,
            }

            if is_numeric:
                stats.update({
                    "mean":   DataProfiler._fmt(series.mean()),
                    "std":    DataProfiler._fmt(series.std()),
                    "min":    DataProfiler._fmt(series.min()),
                    "max":    DataProfiler._fmt(series.max()),
                    "median": DataProfiler._fmt(series.median()),
                })
            else:
                top_val = series.value_counts()
                stats["top"]  = str(top_val.index[0])[:30] if not top_val.empty else "—"
                stats["freq"] = int(top_val.iloc[0])        if not top_val.empty else 0

            columns[col] = stats

        total_missing = int(df.isnull().sum().sum())
        numeric_cols  = int(df.select_dtypes(include="number").shape[1])

        meta = {
            "rows":          len(df),
            "cols":          len(df.columns),
            "numeric_cols":  numeric_cols,
            "cat_cols":      len(df.columns) - numeric_cols,
            "total_missing": total_missing,
            "engine":        "pandas",
        }

        return {"meta": meta, "columns": columns}

    # ------------------------------------------------------------------ #
    #  DuckDB path  — everything is SQL, nothing loads into RAM           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _profile_duckdb(con) -> dict:
        """
        Profile a large Parquet file entirely through DuckDB SQL.
        Each column gets its own aggregation query so we can handle
        mixed types gracefully.
        """
        col_names = AppState.get_column_names()
        col_types = AppState.get_column_types()
        row_count = AppState.get_row_count()

        columns      = {}
        total_missing = 0

        NUMERIC_DUCK_TYPES = {
            "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT",
            "FLOAT", "DOUBLE", "DECIMAL", "REAL", "UBIGINT",
        }

        numeric_col_count = 0

        for col in col_names:
            raw_type   = col_types.get(col, "VARCHAR").upper()
            is_numeric = any(t in raw_type for t in NUMERIC_DUCK_TYPES)

            if is_numeric:
                numeric_col_count += 1

            safe_col = f'"{col}"'

            # Core stats — always safe
            base_sql = f"""
                SELECT
                    COUNT({safe_col})                                    AS cnt,
                    COUNT(DISTINCT {safe_col})                           AS uniq,
                    COUNT(*) - COUNT({safe_col})                         AS missing
                FROM dataset
            """
            base = con.execute(base_sql).fetchone()
            cnt, uniq, missing = int(base[0]), int(base[1]), int(base[2])
            missing_pct = round(missing / row_count * 100, 1) if row_count > 0 else 0
            total_missing += missing

            stats = {
                "dtype":          raw_type,
                "dtype_category": "numeric" if is_numeric else "categorical",
                "count":          cnt,
                "unique":         uniq,
                "missing":        missing,
                "missing_pct":    missing_pct,
            }

            if is_numeric:
                num_sql = f"""
                    SELECT
                        AVG({safe_col})                         AS mean,
                        STDDEV({safe_col})                      AS std,
                        MIN({safe_col})                         AS mn,
                        MAX({safe_col})                         AS mx,
                        MEDIAN({safe_col})                      AS med
                    FROM dataset
                """
                row = con.execute(num_sql).fetchone()
                stats.update({
                    "mean":   DataProfiler._fmt(row[0]),
                    "std":    DataProfiler._fmt(row[1]),
                    "min":    DataProfiler._fmt(row[2]),
                    "max":    DataProfiler._fmt(row[3]),
                    "median": DataProfiler._fmt(row[4]),
                })
            else:
                top_sql = f"""
                    SELECT {safe_col}, COUNT(*) AS freq
                    FROM dataset
                    WHERE {safe_col} IS NOT NULL
                    GROUP BY {safe_col}
                    ORDER BY freq DESC
                    LIMIT 1
                """
                top_row = con.execute(top_sql).fetchone()
                stats["top"]  = str(top_row[0])[:30] if top_row else "—"
                stats["freq"] = int(top_row[1])       if top_row else 0

            columns[col] = stats

        meta = {
            "rows":          row_count,
            "cols":          len(col_names),
            "numeric_cols":  numeric_col_count,
            "cat_cols":      len(col_names) - numeric_col_count,
            "total_missing": total_missing,
            "engine":        "duckdb",          # shown in the UI summary bar
        }

        return {"meta": meta, "columns": columns}

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fmt(val) -> str:
        if val is None:
            return "—"
        if isinstance(val, float) and np.isnan(val):
            return "—"
        if isinstance(val, float):
            return f"{val:,.2f}"
        return str(val)
