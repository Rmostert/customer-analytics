"""
Segmentation engine — two independent methods:

  K-Means  — free-form clustering on any mix of numeric/categorical features.
             User chooses number of clusters (2-20).
             Categorical columns are dummy-encoded on the fly.

  RFM      — pure quartile tiering, NO clustering.
             Each of Recency, Frequency, Monetary gets 4 named tiers:
               R-Tier-1 (most recent) … R-Tier-4 (least recent)
               F-Tier-1 (least frequent) … F-Tier-4 (most frequent)
               M-Tier-1 (lowest spend) … M-Tier-4 (highest spend)
             The combined segment label is e.g. "R-Tier-1 | F-Tier-3 | M-Tier-2".
             A model bundle is still exported so new data can be scored with the
             same quartile cut-points.

Pure Python/pandas/sklearn — no Qt, no AppState.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes

from app.core.duckdb_sample import (
    DEFAULT_BATCH_SIZE,
    fetch_quantile_bins,
    fetch_sample,
    score_in_batches,
)


CLUSTER_LABEL = "cluster_label"   # shared output column for all clustering methods


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class SegmentationResult:
    """Everything the UI needs after a successful segmentation run."""
    method:          str            # "kmeans" | "rfm"
    n_clusters:      int            # K-Means: user choice; RFM: unique segment count
    customer_id_col: str
    feature_cols:    list[str]      # original columns chosen by user
    label_col:       str            # "cluster_label" | "rfm_segment"
    assignments:     pd.DataFrame  # customer_id + original features + tier cols + label
    profile:         pd.DataFrame  # mean/proportion per segment × feature
    distribution:    pd.Series     # count per segment label
    model_bytes:     bytes         # pickle of the fitted model bundle
    inertia:         Optional[float] = None   # K-Means only
    is_sampled:      bool          = False    # model fit used a DuckDB sample
    sample_size:     Optional[int] = None     # rows used to fit the model
    total_rows:      Optional[int] = None     # rows in the source file
    scored_full:     bool          = False    # assignments cover the full file


# ── Shared helpers ────────────────────────────────────────────────────────────

def _dummy_encode(df: pd.DataFrame,
                  feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categoricals, pass numerics through."""
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cats    = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    parts   = [df[numeric]] if numeric else []
    for c in cats:
        parts.append(pd.get_dummies(df[c], prefix=c, drop_first=False, dtype=float))
    encoded = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    return encoded, list(encoded.columns)


def _build_profile(df: pd.DataFrame,
                   feature_cols: list[str],
                   label_col: str) -> pd.DataFrame:
    """Per-segment mean (numeric) or top-category (categorical)."""
    rows = []
    for seg in sorted(df[label_col].unique(), key=str):
        subset = df[df[label_col] == seg]
        row = {label_col: seg, "n": len(subset)}
        for col in feature_cols:
            if col == label_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                row[col] = round(float(subset[col].mean()), 4)
            else:
                top = subset[col].mode()
                row[col] = top.iloc[0] if not top.empty else "—"
        rows.append(row)
    return pd.DataFrame(rows).set_index(label_col)


def _align_encoded(encoded: pd.DataFrame, enc_cols: list[str]) -> pd.DataFrame:
    """Align dummy-encoded batch columns to the training feature set."""
    return encoded.reindex(columns=enc_cols, fill_value=0)


def _copy_result(
    result: SegmentationResult,
    *,
    assignments: pd.DataFrame | None = None,
    profile: pd.DataFrame | None = None,
    distribution: pd.Series | None = None,
    is_sampled: bool = False,
    sample_size: int | None = None,
    total_rows: int | None = None,
    scored_full: bool = False,
) -> SegmentationResult:
    return SegmentationResult(
        method=result.method,
        n_clusters=result.n_clusters,
        customer_id_col=result.customer_id_col,
        feature_cols=result.feature_cols,
        label_col=result.label_col,
        assignments=assignments if assignments is not None else result.assignments,
        profile=profile if profile is not None else result.profile,
        distribution=distribution if distribution is not None else result.distribution,
        model_bytes=result.model_bytes,
        inertia=result.inertia,
        is_sampled=is_sampled,
        sample_size=sample_size,
        total_rows=total_rows,
        scored_full=scored_full,
    )


def _score_kmeans_gmm_batch(batch: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    customer_id_col = bundle["customer_id_col"]
    feature_cols = bundle["feature_cols"]
    enc_cols = bundle["enc_cols"]
    scaler = bundle["scaler"]
    model = bundle["model"]

    work = batch[[customer_id_col] + feature_cols].dropna(subset=feature_cols).copy()
    if work.empty:
        return work

    encoded, _ = _dummy_encode(work, feature_cols)
    encoded = _align_encoded(encoded, enc_cols)
    labels = model.predict(scaler.transform(encoded))
    work[CLUSTER_LABEL] = labels
    return work[[customer_id_col] + feature_cols + [CLUSTER_LABEL]]


def _score_kprototypes_batch(batch: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    customer_id_col = bundle["customer_id_col"]
    feature_cols = bundle["feature_cols"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]
    scaler = bundle.get("scaler")
    model = bundle["model"]

    work = batch[[customer_id_col] + feature_cols].dropna(subset=feature_cols).copy()
    if work.empty:
        return work

    if numeric_cols and scaler is not None:
        scaled = scaler.transform(work[numeric_cols])
    else:
        scaled = np.empty((len(work), 0))

    if categorical_cols:
        cat_data = work[categorical_cols].to_numpy()
        matrix = np.hstack((scaled, cat_data)).astype(object)
    else:
        matrix = scaled

    cat_indices = list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))
    labels = model.predict(matrix, categorical=cat_indices)
    work[CLUSTER_LABEL] = labels
    return work[[customer_id_col] + feature_cols + [CLUSTER_LABEL]]


def _finalize_large_clustering(
    result: SegmentationResult,
    filepath: str,
    customer_id_col: str,
    feature_cols: list[str],
    total_rows: int,
    sample_rows: int,
    score_full: bool,
    batch_size: int,
    progress: Callable[[int], None] | None,
    encoding: str = "utf-8",
) -> SegmentationResult:
    bundle = pickle.loads(result.model_bytes)
    bundle["customer_id_col"] = customer_id_col

    if score_full and total_rows > sample_rows:
        cols = [customer_id_col] + feature_cols
        if result.method == "KPrototypes":
            score_fn = lambda b: _score_kprototypes_batch(b, bundle)
        else:
            score_fn = lambda b: _score_kmeans_gmm_batch(b, bundle)

        assignments = score_in_batches(
            filepath, cols, customer_id_col, total_rows,
            score_fn, batch_size, progress, encoding,
        )
        profile = _build_profile(assignments, feature_cols, CLUSTER_LABEL)
        distribution = assignments[CLUSTER_LABEL].value_counts().sort_index()
        return _copy_result(
            result,
            assignments=assignments,
            profile=profile,
            distribution=distribution,
            is_sampled=True,
            sample_size=sample_rows,
            total_rows=total_rows,
            scored_full=True,
        )

    return _copy_result(
        result,
        is_sampled=True,
        sample_size=sample_rows,
        total_rows=total_rows,
        scored_full=False,
    )


# ── Engine ────────────────────────────────────────────────────────────────────

class SegmentationEngine:

    # ------------------------------------------------------------------ #
    #  K-Means                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def run_kmeans(
        df: pd.DataFrame,
        customer_id_col: str,
        feature_cols: list[str],
        n_clusters: int,
        random_state: int = 42,
    ) -> SegmentationResult:

        work = df[[customer_id_col] + feature_cols].dropna(subset=feature_cols).copy()

        encoded, enc_cols = _dummy_encode(work, feature_cols)
        scaler  = StandardScaler()
        X       = scaler.fit_transform(encoded)

        kmeans  = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels  = kmeans.fit_predict(X)

        work[CLUSTER_LABEL] = labels

        profile      = _build_profile(work, feature_cols, CLUSTER_LABEL)
        distribution = work[CLUSTER_LABEL].value_counts().sort_index()
        assignments  = work[[customer_id_col] + feature_cols + [CLUSTER_LABEL]].copy()

        bundle = {
            "method":       "kmeans",
            "scaler":       scaler,
            "model":        kmeans,
            "feature_cols": feature_cols,
            "enc_cols":     enc_cols,
        }

        return SegmentationResult(
            method=          "kmeans",
            n_clusters=      n_clusters,
            customer_id_col= customer_id_col,
            feature_cols=    feature_cols,
            label_col=       CLUSTER_LABEL,
            assignments=     assignments,
            profile=         profile,
            distribution=    distribution,
            model_bytes=     pickle.dumps(bundle),
            inertia=         float(kmeans.inertia_),
        )

    @staticmethod
    def run_gmm(
        df: pd.DataFrame,
        customer_id_col: str,
        feature_cols: list[str],
        n_clusters: int,
        random_state: int = 42,
    ) -> SegmentationResult:

        work = df[[customer_id_col] + feature_cols].dropna(subset=feature_cols).copy()

        encoded, enc_cols = _dummy_encode(work, feature_cols)
        scaler  = StandardScaler()
        X       = scaler.fit_transform(encoded)

        gmm  = GaussianMixture(n_components=n_clusters, init_params='k-means++', tol=1e-9,random_state=random_state)
        labels  = gmm.fit_predict(X)

        work[CLUSTER_LABEL] = labels

        profile      = _build_profile(work, feature_cols, CLUSTER_LABEL)
        distribution = work[CLUSTER_LABEL].value_counts().sort_index()
        assignments  = work[[customer_id_col] + feature_cols + [CLUSTER_LABEL]].copy()

        bundle = {
            "method":       "GaussianMixture",
            "scaler":       scaler,
            "model":        gmm,
            "feature_cols": feature_cols,
            "enc_cols":     enc_cols,
        }

        return SegmentationResult(
            method=          "GaussianMixture",
            n_clusters=      n_clusters,
            customer_id_col= customer_id_col,
            feature_cols=    feature_cols,
            label_col=       CLUSTER_LABEL,
            assignments=     assignments,
            profile=         profile,
            distribution=    distribution,
            model_bytes=     pickle.dumps(bundle),
            inertia=         None,
        )

    @staticmethod
    def run_kprototypes(
        df: pd.DataFrame,
        customer_id_col: str,
        feature_cols: list[str],
        n_clusters: int,
        random_state: int = 42,
    ) -> SegmentationResult:

        # 1. Isolate features and drop missing values safely
        all_cols = [customer_id_col] + feature_cols
        work = df[all_cols].dropna(subset=feature_cols).copy()
        
        # 2. Identify numeric vs categorical columns cleanly using pandas
        features_df = work[feature_cols]
        numeric_cols = features_df.select_dtypes(include='number').columns.tolist()
        categorical_cols = features_df.select_dtypes(exclude='number').columns.tolist()
        
        # 3. Scale numeric data
        scaler = None
        if numeric_cols:
            scaler = StandardScaler()
            scaled_numeric = scaler.fit_transform(work[numeric_cols])
        else:
            scaled_numeric = np.empty((len(work), 0))
            
        # 4. Combine data into an 'object' array to preserve string/float separation
        if categorical_cols:
            categorical_data = work[categorical_cols].to_numpy()
            data_matrix = np.hstack((scaled_numeric, categorical_data)).astype(object)
        else:
            data_matrix = scaled_numeric
            
        # 5. Map explicit integer positions for categorical columns
        # Because we stacked numeric first, categorical indices start right after them
        cat_indices = list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))

        # 6. Fit the model
        kproto = KPrototypes(n_clusters=n_clusters, init='Cao', random_state=random_state)
        labels = kproto.fit_predict(data_matrix, categorical=cat_indices)

        work[CLUSTER_LABEL] = labels

        profile      = _build_profile(work, feature_cols, CLUSTER_LABEL)
        distribution = work[CLUSTER_LABEL].value_counts().sort_index()
        assignments  = work[[customer_id_col] + feature_cols + [CLUSTER_LABEL]].copy()

        bundle = {
            "method":       "KPrototypes",
            "scaler":       scaler,
            "model":        kproto,
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        }

        return SegmentationResult(
            method=          "KPrototypes",
            n_clusters=      n_clusters,
            customer_id_col= customer_id_col,
            feature_cols=    feature_cols,
            label_col=       CLUSTER_LABEL,
            assignments=     assignments,
            profile=         profile,
            distribution=    distribution,
            model_bytes=     pickle.dumps(bundle),
            inertia=         None,
        )



    # ------------------------------------------------------------------ #
    #  RFM — pure quartile tiering, no K-Means                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def run_rfm(
        df: pd.DataFrame,
        customer_id_col: str,
        recency_col: str,
        frequency_col: str,
        monetary_col: str,
    ) -> SegmentationResult:
        """
        Assign each customer to one of 4 tiers per dimension using quartiles.

        Tier naming:
          Recency   — R-Tier-1 = most recent   (lowest raw value)
                      R-Tier-4 = least recent  (highest raw value)
          Frequency — F-Tier-1 = least frequent (lowest raw value)
                      F-Tier-4 = most frequent  (highest raw value)
          Monetary  — M-Tier-1 = lowest spend   (lowest raw value)
                      M-Tier-4 = highest spend  (highest raw value)

        The combined segment label: "R-Tier-X | F-Tier-Y | M-Tier-Z"

        Cut-points are stored in the model bundle so new rows can be
        scored with the identical boundaries.
        """
        feature_cols = [recency_col, frequency_col, monetary_col]
        work = df[[customer_id_col] + feature_cols].dropna(subset=feature_cols).copy()

        # Convert datefield to date format if text

        if not is_numeric_dtype(df[recency_col]):
            work[recency_col] = pd.to_datetime(work[recency_col])

        # ── Quartile cut-points ──────────────────────────────────────
        r_bins = SegmentationEngine._quartile_bins(work[recency_col])
        f_bins = SegmentationEngine._quartile_bins(work[frequency_col])
        m_bins = SegmentationEngine._quartile_bins(work[monetary_col])

        # ── Tier assignment ──────────────────────────────────────────
        # Recency is INVERTED: lower value → higher (better) tier
        work["R_tier"] = SegmentationEngine._assign_tiers(
            work[recency_col], r_bins, prefix="R", invert=True)

        work["F_tier"] = SegmentationEngine._assign_tiers(
            work[frequency_col], f_bins, prefix="F", invert=False)

        work["M_tier"] = SegmentationEngine._assign_tiers(
            work[monetary_col], m_bins, prefix="M", invert=False)

        # ── Combined segment label ───────────────────────────────────
        work["rfm_segment"] = (
            work["R_tier"] + " | " + work["F_tier"] + " | " + work["M_tier"]
        )

        # ── Profile & distribution ───────────────────────────────────
        tier_cols    = ["R_tier", "F_tier", "M_tier"]
        all_feat     = feature_cols + tier_cols
        profile      = _build_profile(work, all_feat, "rfm_segment")
        distribution = work["rfm_segment"].value_counts().sort_index()

        export_cols  = [customer_id_col] + feature_cols + tier_cols + ["rfm_segment"]
        assignments  = work[export_cols].copy()

        # ── Model bundle (stores cut-points for scoring new data) ────
        bundle = {
            "method":         "rfm",
            "recency_col":    recency_col,
            "frequency_col":  frequency_col,
            "monetary_col":   monetary_col,
            "r_bins":         r_bins,
            "f_bins":         f_bins,
            "m_bins":         m_bins,
        }

        return SegmentationResult(
            method=          "rfm",
            n_clusters=      int(work["rfm_segment"].nunique()),
            customer_id_col= customer_id_col,
            feature_cols=    all_feat,
            label_col=       "rfm_segment",
            assignments=     assignments,
            profile=         profile,
            distribution=    distribution,
            model_bytes=     pickle.dumps(bundle),
            inertia=         None,
        )

    # ------------------------------------------------------------------ #
    #  RFM helpers                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _quartile_bins(series: pd.Series) -> list[float]:
        """
        Return [min, Q1, Q2, Q3, max] cut-points for a series.
        Handles ties gracefully by de-duplicating boundaries.
        """
        qs = series.quantile([0.0, 0.25, 0.50, 0.75, 1.0]).tolist()
        # Ensure strictly increasing (ties collapse adjacent bins)
        result = [qs[0]]
        for v in qs[1:]:
            if v > result[-1]:
                result.append(v)
        return result

    @staticmethod
    def _assign_tiers(series: pd.Series,
                      bins: list[float],
                      prefix: str,
                      invert: bool) -> pd.Series:
        """
        Cut series into up to 4 tiers using the provided bin edges.
        Labels are "{prefix}-Tier-1" … "{prefix}-Tier-4".
        When invert=True the label order is reversed (Tier-1 = highest raw value range).
        """
        n_bins = len(bins) - 1           # number of distinct intervals
        n_tiers = min(n_bins, 4)

        raw_labels = [str(i + 1) for i in range(n_tiers)]

        tier_series = pd.cut(
            series,
            bins=bins,
            labels=raw_labels,
            include_lowest=True,
            duplicates="drop",
        ).astype(str)

        if invert:
            # flip: "1" → "4", "2" → "3", etc.
            max_t = n_tiers
            tier_series = tier_series.map(
                lambda t: str(max_t - int(t) + 1) if t.isdigit() else t
            )

        return prefix + "-Tier-" + tier_series

    # ------------------------------------------------------------------ #
    #  Large CSV / Parquet (DuckDB) — sample fit + optional full scoring   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def run_kmeans_large(
        filepath: str,
        customer_id_col: str,
        feature_cols: list[str],
        n_clusters: int,
        total_rows: int,
        sample_size: int,
        score_full: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        random_state: int = 42,
        progress: Callable[[int], None] | None = None,
        encoding: str = "utf-8",
    ) -> SegmentationResult:
        cols = [customer_id_col] + feature_cols
        sample_df = fetch_sample(filepath, cols, sample_size, encoding=encoding)
        if len(sample_df) < n_clusters:
            raise ValueError(
                f"Sample returned {len(sample_df)} rows — need at least {n_clusters} "
                "non-null rows for clustering. Increase sample size or check missing values."
            )
        result = SegmentationEngine.run_kmeans(
            sample_df, customer_id_col, feature_cols, n_clusters, random_state,
        )
        if progress:
            progress(20)
        return _finalize_large_clustering(
            result, filepath, customer_id_col, feature_cols,
            total_rows, len(sample_df), score_full, batch_size, progress, encoding,
        )

    @staticmethod
    def run_gmm_large(
        filepath: str,
        customer_id_col: str,
        feature_cols: list[str],
        n_clusters: int,
        total_rows: int,
        sample_size: int,
        score_full: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        random_state: int = 42,
        progress: Callable[[int], None] | None = None,
        encoding: str = "utf-8",
    ) -> SegmentationResult:
        cols = [customer_id_col] + feature_cols
        sample_df = fetch_sample(filepath, cols, sample_size, encoding=encoding)
        if len(sample_df) < n_clusters:
            raise ValueError(
                f"Sample returned {len(sample_df)} rows — need at least {n_clusters} "
                "non-null rows for clustering. Increase sample size or check missing values."
            )
        result = SegmentationEngine.run_gmm(
            sample_df, customer_id_col, feature_cols, n_clusters, random_state,
        )
        if progress:
            progress(20)
        return _finalize_large_clustering(
            result, filepath, customer_id_col, feature_cols,
            total_rows, len(sample_df), score_full, batch_size, progress, encoding,
        )

    @staticmethod
    def run_kprototypes_large(
        filepath: str,
        customer_id_col: str,
        feature_cols: list[str],
        n_clusters: int,
        total_rows: int,
        sample_size: int,
        score_full: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        random_state: int = 42,
        progress: Callable[[int], None] | None = None,
        encoding: str = "utf-8",
    ) -> SegmentationResult:
        cols = [customer_id_col] + feature_cols
        sample_df = fetch_sample(filepath, cols, sample_size, encoding=encoding)
        if len(sample_df) < n_clusters:
            raise ValueError(
                f"Sample returned {len(sample_df)} rows — need at least {n_clusters} "
                "non-null rows for clustering. Increase sample size or check missing values."
            )
        result = SegmentationEngine.run_kprototypes(
            sample_df, customer_id_col, feature_cols, n_clusters, random_state,
        )
        if progress:
            progress(20)
        return _finalize_large_clustering(
            result, filepath, customer_id_col, feature_cols,
            total_rows, len(sample_df), score_full, batch_size, progress, encoding,
        )

    @staticmethod
    def run_rfm_large(
        filepath: str,
        customer_id_col: str,
        recency_col: str,
        frequency_col: str,
        monetary_col: str,
        total_rows: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress: Callable[[int], None] | None = None,
        encoding: str = "utf-8",
    ) -> SegmentationResult:
        """
        RFM on a large CSV / Parquet file: quartiles from the full dataset via DuckDB,
        tier assignment streamed in batches (no sampling).
        """
        feature_cols = [recency_col, frequency_col, monetary_col]
        cols = [customer_id_col] + feature_cols

        r_bins = fetch_quantile_bins(filepath, recency_col, encoding=encoding)
        f_bins = fetch_quantile_bins(filepath, frequency_col, encoding=encoding)
        m_bins = fetch_quantile_bins(filepath, monetary_col, encoding=encoding)

        bundle = {
            "method":        "rfm",
            "recency_col":   recency_col,
            "frequency_col": frequency_col,
            "monetary_col":  monetary_col,
            "r_bins":        r_bins,
            "f_bins":        f_bins,
            "m_bins":        m_bins,
        }

        def score_rfm_batch(batch: pd.DataFrame) -> pd.DataFrame:
            work = batch[cols].dropna(subset=feature_cols).copy()
            if work.empty:
                return work
            if not is_numeric_dtype(work[recency_col]):
                work[recency_col] = pd.to_datetime(work[recency_col])
            work["R_tier"] = SegmentationEngine._assign_tiers(
                work[recency_col], r_bins, prefix="R", invert=True)
            work["F_tier"] = SegmentationEngine._assign_tiers(
                work[frequency_col], f_bins, prefix="F", invert=False)
            work["M_tier"] = SegmentationEngine._assign_tiers(
                work[monetary_col], m_bins, prefix="M", invert=False)
            work["rfm_segment"] = (
                work["R_tier"] + " | " + work["F_tier"] + " | " + work["M_tier"]
            )
            tier_cols = ["R_tier", "F_tier", "M_tier"]
            return work[[customer_id_col] + feature_cols + tier_cols + ["rfm_segment"]]

        assignments = score_in_batches(
            filepath, cols, customer_id_col, total_rows,
            score_rfm_batch, batch_size, progress, encoding,
        )
        if assignments.empty:
            raise ValueError("No rows with complete R/F/M values found in the dataset.")

        tier_cols = ["R_tier", "F_tier", "M_tier"]
        all_feat = feature_cols + tier_cols
        profile = _build_profile(assignments, all_feat, "rfm_segment")
        distribution = assignments["rfm_segment"].value_counts().sort_index()

        return SegmentationResult(
            method="rfm",
            n_clusters=int(assignments["rfm_segment"].nunique()),
            customer_id_col=customer_id_col,
            feature_cols=all_feat,
            label_col="rfm_segment",
            assignments=assignments,
            profile=profile,
            distribution=distribution,
            model_bytes=pickle.dumps(bundle),
            inertia=None,
            is_sampled=False,
            sample_size=None,
            total_rows=total_rows,
            scored_full=True,
        )
