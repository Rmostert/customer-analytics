"""
Segmentation page — K-Means clustering and RFM tiering.

K-Means: user picks features + number of clusters.
RFM:     user picks R/F/M columns only; 4 tiers per dimension, no cluster count needed.
"""

from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QSpinBox, QListWidget, QAbstractItemView,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QProgressBar, QSplitter, QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont

from app.utils.app_state import AppState
from app.core.segmentation import SegmentationEngine, SegmentationResult
from app.core.duckdb_sample import DEFAULT_SAMPLE_SIZE, MAX_SAMPLE_SIZE

import pandas as pd


# ── Palette ───────────────────────────────────────────────────────────────────

CLUSTER_PALETTE = [
    "#4F7CFF", "#3DDC84", "#FF7A45", "#7B5EA7",
    "#FFD166", "#EF476F", "#06D6A0", "#118AB2",
    "#FF6B9D", "#C77DFF", "#4CC9F0", "#F4A261",
    "#2EC4B6", "#E76F51", "#A8DADC", "#457B9D",
]


# ── Worker ────────────────────────────────────────────────────────────────────

class SegWorker(QThread):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self):
        try:
            cfg = self.config
            self.progress.emit(15)
            progress = self.progress.emit
            large = cfg.get("duckdb_filepath")

            if cfg["method"] == "kmeans":
                if large:
                    result = SegmentationEngine.run_kmeans_large(
                        filepath=        cfg["duckdb_filepath"],
                        customer_id_col= cfg["id_col"],
                        feature_cols=    cfg["feature_cols"],
                        n_clusters=      cfg["n_clusters"],
                        total_rows=      cfg["total_rows"],
                        sample_size=     cfg["sample_size"],
                        score_full=      cfg.get("score_full", True),
                        progress=        progress,
                        encoding=        cfg.get("encoding", "utf-8"),
                    )
                else:
                    result = SegmentationEngine.run_kmeans(
                        df=              cfg["df"],
                        customer_id_col= cfg["id_col"],
                        feature_cols=    cfg["feature_cols"],
                        n_clusters=      cfg["n_clusters"],
                    )
            elif cfg["method"] == "GaussianMixture":
                if large:
                    result = SegmentationEngine.run_gmm_large(
                        filepath=        cfg["duckdb_filepath"],
                        customer_id_col= cfg["id_col"],
                        feature_cols=    cfg["feature_cols"],
                        n_clusters=      cfg["n_clusters"],
                        total_rows=      cfg["total_rows"],
                        sample_size=     cfg["sample_size"],
                        score_full=      cfg.get("score_full", True),
                        progress=        progress,
                        encoding=        cfg.get("encoding", "utf-8"),
                    )
                else:
                    result = SegmentationEngine.run_gmm(
                        df=              cfg["df"],
                        customer_id_col= cfg["id_col"],
                        feature_cols=    cfg["feature_cols"],
                        n_clusters=      cfg["n_clusters"],
                    )
            elif cfg["method"] == "KPrototypes":
                if large:
                    result = SegmentationEngine.run_kprototypes_large(
                        filepath=        cfg["duckdb_filepath"],
                        customer_id_col= cfg["id_col"],
                        feature_cols=    cfg["feature_cols"],
                        n_clusters=      cfg["n_clusters"],
                        total_rows=      cfg["total_rows"],
                        sample_size=     cfg["sample_size"],
                        score_full=      cfg.get("score_full", True),
                        progress=        progress,
                        encoding=        cfg.get("encoding", "utf-8"),
                    )
                else:
                    result = SegmentationEngine.run_kprototypes(
                        df=              cfg["df"],
                        customer_id_col= cfg["id_col"],
                        feature_cols=    cfg["feature_cols"],
                        n_clusters=      cfg["n_clusters"],
                    )
            elif large:
                result = SegmentationEngine.run_rfm_large(
                    filepath=        cfg["duckdb_filepath"],
                    customer_id_col= cfg["id_col"],
                    recency_col=     cfg["recency_col"],
                    frequency_col=   cfg["frequency_col"],
                    monetary_col=    cfg["monetary_col"],
                    total_rows=      cfg["total_rows"],
                    progress=        progress,
                    encoding=        cfg.get("encoding", "utf-8"),
                )
            else:
                result = SegmentationEngine.run_rfm(
                    df=              cfg["df"],
                    customer_id_col= cfg["id_col"],
                    recency_col=     cfg["recency_col"],
                    frequency_col=   cfg["frequency_col"],
                    monetary_col=    cfg["monetary_col"],
                )

            self.progress.emit(90)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Distribution chart (pure QPainter) ───────────────────────────────────────

class DistributionChart(QWidget):
    """Horizontal bar chart for segment / cluster sizes."""

    def __init__(self):
        super().__init__()
        self._data: list[tuple[str, int]] = []
        self.setMinimumHeight(40)

    def set_data(self, distribution: pd.Series):
        self._data = [(str(k), int(v)) for k, v in distribution.sort_index().items()]
        # Resize to fit all bars
        bar_h  = 36
        gap    = 6
        margin = 16
        needed = margin * 2 + len(self._data) * (bar_h + gap)
        self.setMinimumHeight(max(needed, 80))
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return

        p      = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        total   = sum(v for _, v in self._data)
        w       = self.width()
        margin  = 16
        bar_h   = 32
        label_w = 230      # wide enough for "R-Tier-1 | F-Tier-2 | M-Tier-3"
        count_w = 120
        bar_max = w - label_w - count_w - margin * 2

        y = margin
        for i, (label, count) in enumerate(self._data):
            color = QColor(CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)])
            pct   = count / total if total else 0
            bar_w = max(int(bar_max * pct), 4)

            # Label
            p.setPen(QColor("#7A7F9A"))
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(margin, y, label_w - 8, bar_h,
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       label)

            # Bar
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            bx = margin + label_w
            p.drawRoundedRect(bx, y + 4, bar_w, bar_h - 8, 4, 4)

            # Count + pct
            p.setPen(QColor("#E8EAF2"))
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(bx + bar_w + 8, y, count_w, bar_h,
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       f"{count:,}  ({pct * 100:.1f}%)")

            y += bar_h + 6

        p.end()


# ── Main page ─────────────────────────────────────────────────────────────────

class SegmentationPage(QWidget):

    def __init__(self):
        super().__init__()
        self._worker: SegWorker | None      = None
        self._result: SegmentationResult | None = None
        self._build_ui()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_columns()
        self._update_large_file_state()

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_config_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setSizes([320, 900])

        root.addWidget(splitter)

    # ── Left config panel ─────────────────────────────────────────────

    def _build_config_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("config_panel")
        panel.setFixedWidth(320)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout  = QVBoxLayout(content)
        layout.setContentsMargins(24, 28, 24, 24)
        layout.setSpacing(16)

        # Title
        title = QLabel("Segmentation")
        title.setObjectName("page_title")
        subtitle = QLabel("Configure and run customer segmentation.")
        subtitle.setObjectName("page_subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self._large_file_banner = QLabel(
            "Large-file mode (DuckDB): clustering fits on a random sample, then "
            "optionally assigns labels to the full file in batches. "
            "RFM uses the full dataset for quartiles and tier assignment."
        )
        self._large_file_banner.setObjectName("status_label")
        self._large_file_banner.setWordWrap(True)
        self._large_file_banner.setVisible(False)
        layout.addWidget(self._large_file_banner)

        self._large_file_box = QWidget()
        self._large_file_box.setVisible(False)
        lf = QVBoxLayout(self._large_file_box)
        lf.setContentsMargins(0, 0, 0, 0)
        lf.setSpacing(8)

        lf.addWidget(self._section("Sample size  (clustering methods)"))
        sample_row = QHBoxLayout()
        self._sample_spin = QSpinBox()
        self._sample_spin.setRange(1_000, MAX_SAMPLE_SIZE)
        self._sample_spin.setValue(DEFAULT_SAMPLE_SIZE)
        self._sample_spin.setSingleStep(5_000)
        self._sample_spin.setFixedWidth(120)
        sample_row.addWidget(self._sample_spin)
        sample_row.addStretch()
        lf.addLayout(sample_row)

        self._score_full_chk = QCheckBox("Assign clusters to full dataset")
        self._score_full_chk.setChecked(True)
        self._score_full_chk.setToolTip(
            "After fitting on the sample, predict cluster labels for all rows via DuckDB batches."
        )
        lf.addWidget(self._score_full_chk)

        layout.addWidget(self._large_file_box)

        # Method selector
        layout.addWidget(self._section("Method"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["K-Means Clustering", "Gaussian mixture Clustering","K Prototypes", "RFM Segmentation"])
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        layout.addWidget(self._method_combo)

        # Customer ID
        layout.addWidget(self._section("Customer ID Column"))
        self._id_combo = QComboBox()
        layout.addWidget(self._id_combo)

        # ── K-Means block ─────────────────────────────────────────────
        self._kmeans_box = QWidget()
        km = QVBoxLayout(self._kmeans_box)
        km.setContentsMargins(0, 0, 0, 0)
        km.setSpacing(10)

        km.addWidget(self._section("Number of Clusters"))
        self._n_spin = QSpinBox()
        self._n_spin.setRange(2, 20)
        self._n_spin.setValue(4)
        self._n_spin.setFixedWidth(80)
        km.addWidget(self._n_spin)

        km.addWidget(self._section("Feature Columns  (select ≥ 2)"))
        self._feature_list = QListWidget()
        self._feature_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        self._feature_list.setFixedHeight(180)
        km.addWidget(self._feature_list)

        note = QLabel("Categorical columns are dummy-encoded automatically.")
        note.setObjectName("field_label")
        note.setWordWrap(True)
        km.addWidget(note)

        layout.addWidget(self._kmeans_box)

        # ── RFM block ─────────────────────────────────────────────────
        self._rfm_box = QWidget()
        rfm = QVBoxLayout(self._rfm_box)
        rfm.setContentsMargins(0, 0, 0, 0)
        rfm.setSpacing(10)

        rfm_info = QLabel(
            "Each dimension is split into 4 tiers using quartiles.\n"
            "No cluster count needed."
        )
        rfm_info.setObjectName("field_label")
        rfm_info.setWordWrap(True)
        rfm.addWidget(rfm_info)

        # Tier legend
        legend = QFrame()
        legend.setObjectName("rfm_legend")
        leg_layout = QVBoxLayout(legend)
        leg_layout.setContentsMargins(10, 8, 10, 8)
        leg_layout.setSpacing(2)
        for line in [
            "R-Tier-1  most recent  →  R-Tier-4  least recent",
            "F-Tier-1  least frequent  →  F-Tier-4  most frequent",
            "M-Tier-1  lowest spend  →  M-Tier-4  highest spend",
        ]:
            lbl = QLabel(line)
            lbl.setObjectName("rfm_legend_line")
            lbl.setWordWrap(True)
            leg_layout.addWidget(lbl)
        rfm.addWidget(legend)

        for attr, label, tip in [
            ("_rfm_recency",   "Recency Column",   "Lower = more recent (e.g. days since last order)"),
            ("_rfm_frequency", "Frequency Column", "Higher = more frequent (e.g. order count)"),
            ("_rfm_monetary",  "Monetary Column",  "Higher = more valuable (e.g. total spend)"),
        ]:
            rfm.addWidget(self._section(label))
            combo = QComboBox()
            combo.setToolTip(tip)
            setattr(self, attr, combo)
            rfm.addWidget(combo)

        layout.addWidget(self._rfm_box)
        self._rfm_box.setVisible(False)

        # Run
        self._run_btn = QPushButton("▶  Run Segmentation")
        self._run_btn.setObjectName("primary_btn")
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setObjectName("load_progress")
        self._progress.setVisible(False)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setObjectName("status_label")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        layout.addStretch()

        # Export
        self._export_frame = QWidget()
        self._export_frame.setVisible(False)
        exp = QVBoxLayout(self._export_frame)
        exp.setContentsMargins(0, 0, 0, 0)
        exp.setSpacing(8)
        exp.addWidget(self._section("Export"))

        self._exp_excel_btn   = QPushButton("⬇  Assignments (.xlsx)")
        self._exp_model_btn   = QPushButton("⬇  Model (.pkl)")

        for btn in [self._exp_excel_btn, self._exp_model_btn]:
            btn.setObjectName("secondary_btn")
            exp.addWidget(btn)

        self._exp_excel_btn.clicked.connect(lambda: self._export_assignments("xlsx"))
        self._exp_model_btn.clicked.connect(self._export_model)

        layout.addWidget(self._export_frame)

        scroll.setWidget(content)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        return panel

    # ── Right results panel ───────────────────────────────────────────

    def _build_results_panel(self) -> QWidget:
        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(20)

        self._placeholder = QLabel(
            "Configure settings and click  ▶ Run Segmentation  to see results.")
        self._placeholder.setObjectName("page_subtitle")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._placeholder, alignment=Qt.AlignmentFlag.AlignCenter)

        self._results_widget = QWidget()
        self._results_widget.setVisible(False)
        res = QVBoxLayout(self._results_widget)
        res.setContentsMargins(0, 0, 0, 0)
        res.setSpacing(20)

        # Summary strip
        self._summary_strip = QFrame()
        self._summary_strip.setObjectName("summary_bar")
        strip = QHBoxLayout(self._summary_strip)
        strip.setContentsMargins(16, 8, 16, 8)
        self._lbl_method   = self._strip_lbl("Method",   "—")
        self._lbl_segments = self._strip_lbl("Segments", "—")
        self._lbl_rows     = self._strip_lbl("Rows",     "—")
        self._lbl_inertia  = self._strip_lbl("Inertia",  "—")
        for lbl in [self._lbl_method, self._lbl_segments,
                    self._lbl_rows, self._lbl_inertia]:
            strip.addWidget(lbl)
        res.addWidget(self._summary_strip)

        # Distribution chart inside a scroll area (RFM can have many segments)
        dist_lbl = QLabel("Segment Distribution")
        dist_lbl.setObjectName("section_label")
        res.addWidget(dist_lbl)

        chart_scroll = QScrollArea()
        chart_scroll.setWidgetResizable(True)
        chart_scroll.setFrameShape(QFrame.Shape.NoFrame)
        chart_scroll.setFixedHeight(260)

        self._chart = DistributionChart()
        chart_scroll.setWidget(self._chart)
        res.addWidget(chart_scroll)

        # Profile table
        profile_lbl = QLabel("Segment Profiles  (mean / top category per feature)")
        profile_lbl.setObjectName("section_label")
        res.addWidget(profile_lbl)

        self._profile_table = QTableWidget()
        self._profile_table.setObjectName("preview_table")
        self._profile_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._profile_table.setAlternatingRowColors(True)
        self._profile_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._profile_table.setMinimumHeight(220)
        res.addWidget(self._profile_table)

        layout.addWidget(self._results_widget)
        return panel

    # ── Helpers ───────────────────────────────────────────────────────

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("section_label")
        return lbl

    def _strip_lbl(self, key: str, value: str) -> QLabel:
        lbl = QLabel(f"{key}: {value}")
        lbl.setObjectName("summary_item")
        return lbl

    # ------------------------------------------------------------------ #
    #  Column population                                                   #
    # ------------------------------------------------------------------ #

    def _refresh_columns(self):
        cols = AppState.get_column_names()
        if not cols:
            return
        for combo in [self._id_combo, self._rfm_recency,
                      self._rfm_frequency, self._rfm_monetary]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(cols)
            combo.blockSignals(False)
        self._feature_list.clear()
        for col in cols:
            self._feature_list.addItem(col)

    def _on_method_changed(self, idx: int):
        self._kmeans_box.setVisible(idx < 3)
        self._rfm_box.setVisible(idx == 3)
        if AppState.is_large():
            is_rfm = idx == 3
            self._large_file_box.setVisible(not is_rfm)
            self._large_file_banner.setText(
                "Large-file mode (DuckDB): RFM computes quartiles and assigns tiers "
                "across the full dataset in batches."
                if is_rfm else
                "Large-file mode (DuckDB): clustering fits on a random sample, then "
                "optionally assigns labels to the full file in batches."
            )

    def _update_large_file_state(self):
        is_large = AppState.is_large()
        self._large_file_banner.setVisible(is_large)
        if not is_large:
            self._large_file_box.setVisible(False)
            if self._status.text().startswith("❌  Segmentation unavailable"):
                self._status.setText("")
            return

        total = AppState.get_row_count()
        max_sample = max(1_000, min(total, MAX_SAMPLE_SIZE))
        self._sample_spin.setMaximum(max_sample)
        self._sample_spin.setValue(min(DEFAULT_SAMPLE_SIZE, max_sample))

        is_rfm = self._method_combo.currentIndex() == 3
        self._large_file_box.setVisible(not is_rfm)
        self._on_method_changed(self._method_combo.currentIndex())
        self._run_btn.setEnabled(True)
        if not self._status.text().startswith("✅"):
            self._status.setText("")

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #

    def _run(self):
        if self._worker and self._worker.isRunning():
            return

        is_large = AppState.is_large()
        df = AppState.get_dataframe()
        if df is None and not is_large:
            self._status.setText("❌  No dataset loaded. Please import data first.")
            return

        id_col  = self._id_combo.currentText()
        large_kw = {}
        if is_large:
            filepath = AppState.get_filepath()
            if not filepath:
                self._status.setText("❌  Large dataset path not available.")
                return
            large_kw = {
                "duckdb_filepath": filepath,
                "total_rows":      AppState.get_row_count(),
                "sample_size":     self._sample_spin.value(),
                "score_full":      self._score_full_chk.isChecked(),
                "encoding":        AppState.get_load_encoding(),
            }

        if self._method_combo.currentIndex() == 0:
            selected = [self._feature_list.item(i).text()
                        for i in range(self._feature_list.count())
                        if self._feature_list.item(i).isSelected()]
            if len(selected) < 2:
                self._status.setText("❌  Select at least 2 feature columns.")
                return
            config = {
                "method":       "kmeans",
                "df":           df,
                "id_col":       id_col,
                "feature_cols": selected,
                "n_clusters":   self._n_spin.value(),
                **large_kw,
            }

        elif self._method_combo.currentIndex() == 1:

            selected = [self._feature_list.item(i).text()
                        for i in range(self._feature_list.count())
                        if self._feature_list.item(i).isSelected()]
            if len(selected) < 2:
                self._status.setText("❌  Select at least 2 feature columns.")
                return
            config = {
                "method":       "GaussianMixture",
                "df":           df,
                "id_col":       id_col,
                "feature_cols": selected,
                "n_clusters":   self._n_spin.value(),
                **large_kw,
            }

        elif self._method_combo.currentIndex() == 2:

            selected = [self._feature_list.item(i).text()
                        for i in range(self._feature_list.count())
                        if self._feature_list.item(i).isSelected()]
            if len(selected) < 2:
                self._status.setText("❌  Select at least 2 feature columns.")
                return
            config = {
                "method":       "KPrototypes",
                "df":           df,
                "id_col":       id_col,
                "feature_cols": selected,
                "n_clusters":   self._n_spin.value(),
                **large_kw,
            }

        else:
            config = {
                "method":        "rfm",
                "df":            df,
                "id_col":        id_col,
                "recency_col":   self._rfm_recency.currentText(),
                "frequency_col": self._rfm_frequency.currentText(),
                "monetary_col":  self._rfm_monetary.currentText(),
                **large_kw,
            }

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("Running segmentation…")
        self._results_widget.setVisible(False)
        self._placeholder.setVisible(True)
        self._export_frame.setVisible(False)

        self._worker = SegWorker(config)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # ------------------------------------------------------------------ #
    #  Results                                                             #
    # ------------------------------------------------------------------ #

    def _on_done(self, result: SegmentationResult):
        self._result = result
        AppState.set_segmentation_result(result)

        self._progress.setValue(100)
        self._progress.setVisible(False)
        self._update_large_file_state()

        n_seg = result.distribution.nunique()
        status = f"✅  Done — {n_seg} segments, {len(result.assignments):,} rows assigned."
        if result.is_sampled and result.sample_size:
            if result.scored_full:
                status += f"  (model fit on {result.sample_size:,}-row sample)"
            else:
                status += f"  (sample only — {result.sample_size:,} rows)"
        elif result.scored_full and result.total_rows:
            status += f"  (full dataset — {result.total_rows:,} rows)"
        self._status.setText(status)

        # Summary strip
        method_str = result.method
        self._lbl_method.setText(f"Method: {method_str}")
        self._lbl_segments.setText(f"Segments: {n_seg}")
        self._lbl_rows.setText(f"Rows: {len(result.assignments):,}")
        inertia_txt = f"{result.inertia:,.0f}" if result.inertia is not None else "N/A"
        self._lbl_inertia.setText(f"Inertia: {inertia_txt}")

        self._chart.set_data(result.distribution)
        self._populate_profile(result.profile, result.label_col)

        self._placeholder.setVisible(False)
        self._results_widget.setVisible(True)
        self._export_frame.setVisible(True)

    def _populate_profile(self, profile: pd.DataFrame, label_col: str):
        tbl  = self._profile_table
        tbl.clear()

        segs  = list(profile.index)
        cols  = [c for c in profile.columns]
        dcols = ["n"] + [c for c in cols if c != "n"] if "n" in cols else cols

        tbl.setRowCount(len(segs))
        tbl.setColumnCount(len(dcols) + 1)
        tbl.setHorizontalHeaderLabels([label_col] + dcols)

        for r, seg in enumerate(segs):
            color_idx = r % len(CLUSTER_PALETTE)
            seg_item  = QTableWidgetItem(f"  {seg}")
            seg_item.setForeground(QColor(CLUSTER_PALETTE[color_idx]))
            tbl.setItem(r, 0, seg_item)

            for c, col in enumerate(dcols):
                val  = profile.loc[seg, col]
                text = f"{val:,.4f}" if isinstance(val, float) else str(val)
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                tbl.setItem(r, c + 1, cell)

    def _on_error(self, msg: str):
        self._progress.setVisible(False)
        self._update_large_file_state()
        self._status.setText(f"❌  {msg}")
        QMessageBox.critical(self, "Segmentation Error", msg)

    # ------------------------------------------------------------------ #
    #  Export                                                              #
    # ------------------------------------------------------------------ #

    def _export_assignments(self, fmt: str):
        if not self._result:
            return
        filter_ = "Excel Files (*.xlsx)" 
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Assignments", f"segment_assignments.{fmt}", filter_)
        if not path:
            return
        try:
            self._result.assignments.to_excel(path, index=False)
            QMessageBox.information(self, "Export Complete", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_model(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Model", "segmentation_model.pkl",
            "Pickle Files (*.pkl)")
        if not path:
            return
        try:
            with open(path, "wb") as f:
                f.write(self._result.model_bytes)
            QMessageBox.information(self, "Export Complete", f"Model saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
