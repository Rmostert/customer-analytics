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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes


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
            "feature_cols": feature_cols
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
