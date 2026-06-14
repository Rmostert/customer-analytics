"""
DataProfiler — generates per-column statistics for the Explore page.

Two code paths:
  • pandas path  — dataset fits in memory (CSV, Excel, JSON, small Parquet)
  • DuckDB path  — large CSV / Parquet file; all aggregations run as SQL queries
                   so only the results come back into memory, never the raw data.
"""

import pandas as pd
from numbers import Number
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
            if DataProfiler._is_complex_pandas_series(series):
                continue

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

        profiled_df = df[list(columns.keys())]
        total_missing = int(profiled_df.isnull().sum().sum())
        numeric_cols  = int(profiled_df.select_dtypes(include="number").shape[1])

        meta = {
            "rows":          len(df),
            "cols":          len(columns),
            "numeric_cols":  numeric_cols,
            "cat_cols":      len(columns) - numeric_cols,
            "total_missing": total_missing,
            "engine":        "pandas",
        }

        return {"meta": meta, "columns": columns}

    # ------------------------------------------------------------------ #
    #  DuckDB path  — summarize in SQL without loading raw data           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _profile_duckdb(con) -> dict:
        """
        Profile a large CSV / Parquet file through DuckDB's native SUMMARIZE.
        SUMMARIZE returns one metadata row per source column, so the raw
        dataset never comes back into Python.
        """
        row_count = AppState.get_row_count()
        summary_df = con.execute("SUMMARIZE SELECT * FROM dataset").df()

        columns      = {}
        total_missing = 0
        numeric_col_count = 0

        for _, row in summary_df.iterrows():
            col = row["column_name"]
            raw_type = str(row["column_type"]).upper()
            if DataProfiler._is_complex_duck_type(raw_type):
                continue

            is_numeric = DataProfiler._is_numeric_duck_type(raw_type)

            if is_numeric:
                numeric_col_count += 1

            missing_pct = DataProfiler._fmt_pct(row["null_percentage"])
            missing = DataProfiler._missing_from_pct(row_count, missing_pct)
            non_null_count = max(row_count - missing, 0)
            unique = DataProfiler._to_int(row["approx_unique"])
            total_missing += missing

            stats = {
                "dtype":          raw_type,
                "dtype_category": "numeric" if is_numeric else "categorical",
                "count":          non_null_count,
                "unique":         unique,
                "missing":        missing,
                "missing_pct":    missing_pct,
            }

            if is_numeric:
                stats.update({
                    "mean":   DataProfiler._fmt(row["avg"]),
                    "std":    DataProfiler._fmt(row["std"]),
                    "min":    DataProfiler._fmt(row["min"]),
                    "max":    DataProfiler._fmt(row["max"]),
                    "median": DataProfiler._fmt(row["q50"]),
                })
            else:
                safe_col = DataProfiler._quote_identifier(col)
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
            "cols":          len(columns),
            "numeric_cols":  numeric_col_count,
            "cat_cols":      len(columns) - numeric_col_count,
            "total_missing": total_missing,
            "engine":        "duckdb summarize",
        }

        return {"meta": meta, "columns": columns}

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fmt(val) -> str:
        if val is None:
            return "—"
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return "—"
            try:
                val = float(val)
            except ValueError:
                return val
        if isinstance(val, Number) and pd.isna(val):
            return "—"
        if isinstance(val, Number) and not isinstance(val, bool):
            if float(val).is_integer():
                return f"{int(val):,}"
            return f"{val:,.2f}"
        return str(val)

    @staticmethod
    def _fmt_pct(val) -> float:
        if val is None:
            return 0.0
        if isinstance(val, str):
            val = val.replace("%", "").strip()
        if pd.isna(val):
            return 0.0
        return round(float(val), 1)

    @staticmethod
    def _missing_from_pct(row_count: int, missing_pct: float) -> int:
        if row_count <= 0:
            return 0
        return int(round(row_count * missing_pct / 100))

    @staticmethod
    def _to_int(val) -> int:
        if val is None or pd.isna(val):
            return 0
        return int(round(float(val)))

    @staticmethod
    def _is_numeric_duck_type(raw_type: str) -> bool:
        numeric_duck_types = {
            "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT",
            "UINTEGER", "UBIGINT", "USMALLINT", "UTINYINT",
            "FLOAT", "DOUBLE", "DECIMAL", "REAL",
        }
        return any(t in raw_type for t in numeric_duck_types)

    @staticmethod
    def _is_complex_duck_type(raw_type: str) -> bool:
        raw_type = raw_type.upper()
        complex_prefixes = ("STRUCT", "MAP", "UNION", "LIST")
        return raw_type.startswith(complex_prefixes) or "[]" in raw_type

    @staticmethod
    def _is_complex_pandas_series(series: pd.Series) -> bool:
        non_null = series.dropna()
        if non_null.empty:
            return False
        return non_null.map(
            lambda val: isinstance(val, (dict, list, tuple, set))
        ).any()

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'
