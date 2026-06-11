"""
Explore page — column profiles, distributions, and missing value analysis.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QGridLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QSizePolicy, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from app.utils.app_state import AppState
from app.core.profiler import DataProfiler


class ProfileWorker(QThread):
    finished = pyqtSignal(dict, int)
    error    = pyqtSignal(str)

    def __init__(self, dataset_version: int):
        super().__init__()
        self.dataset_version = dataset_version

    def run(self):
        try:
            df = AppState.get_dataframe()
            if df is None:
                self.error.emit("No dataset loaded. Please import data first.")
                return
            profile = DataProfiler.profile(df)
            self.finished.emit(profile, self.dataset_version)
        except Exception as e:
            self.error.emit(str(e))


class ColumnCard(QFrame):
    """Card showing stats for a single column."""

    def __init__(self, col_name: str, stats: dict):
        super().__init__()
        self.setObjectName("column_card")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # Column name + type badge
        header = QHBoxLayout()
        name_lbl = QLabel(col_name)
        name_lbl.setObjectName("card_col_name")

        dtype_lbl = QLabel(stats.get("dtype", ""))
        dtype_lbl.setObjectName("dtype_badge")
        dtype_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        header.addWidget(name_lbl)
        header.addWidget(dtype_lbl)
        layout.addLayout(header)

        # Stats grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(2)

        stat_items = self._build_stat_items(stats)
        for i, (key, val) in enumerate(stat_items):
            row, col = divmod(i, 2)
            key_lbl = QLabel(key)
            key_lbl.setObjectName("stat_key_small")
            val_lbl = QLabel(str(val))
            val_lbl.setObjectName("stat_val_small")
            pair = QHBoxLayout()
            pair.addWidget(key_lbl)
            pair.addStretch()
            pair.addWidget(val_lbl)
            wrapper = QWidget()
            wrapper.setLayout(pair)
            grid.addWidget(wrapper, row, col)

        layout.addLayout(grid)

        # Missing bar
        missing_pct = stats.get("missing_pct", 0)
        bar_row = QHBoxLayout()
        bar_label = QLabel(f"Missing: {missing_pct:.1f}%")
        bar_label.setObjectName("stat_key_small")

        bar_bg = QFrame()
        bar_bg.setObjectName("missing_bar_bg")
        bar_bg.setFixedHeight(6)

        bar_fill = QFrame(bar_bg)
        bar_fill.setObjectName("missing_bar_fill")
        bar_fill.setFixedHeight(6)

        bar_row.addWidget(bar_label)
        bar_row.addWidget(bar_bg, 1)
        layout.addLayout(bar_row)

        # Resize bar after widget is shown
        self._bar_bg   = bar_bg
        self._bar_fill = bar_fill
        self._missing_pct = missing_pct

    def resizeEvent(self, event):
        super().resizeEvent(event)
        fill_w = int(self._bar_bg.width() * self._missing_pct / 100)
        self._bar_fill.setFixedWidth(max(fill_w, 0))

    def _build_stat_items(self, stats: dict) -> list:
        items = [
            ("Count",   f"{stats.get('count', '—'):,}" if isinstance(stats.get('count'), int) else stats.get('count', '—')),
            ("Unique",  f"{stats.get('unique', '—'):,}" if isinstance(stats.get('unique'), int) else stats.get('unique', '—')),
        ]
        if stats.get("dtype_category") == "numeric":
            items += [
                ("Mean",   stats.get("mean",   "—")),
                ("Std",    stats.get("std",    "—")),
                ("Min",    stats.get("min",    "—")),
                ("Max",    stats.get("max",    "—")),
                ("Median", stats.get("median", "—")),
            ]
        else:
            items += [
                ("Top value",  stats.get("top",   "—")),
                ("Freq",       stats.get("freq",  "—")),
            ]
        return items


class ExplorePage(QWidget):
    def __init__(self):
        super().__init__()
        self._worker = None
        self._profiled_version = None
        self._build_ui()

    def showEvent(self, event):
        """Auto-profile when the page becomes visible and data is available."""
        super().showEvent(event)
        if (
            AppState.get_dataframe() is not None
            and self._profiled_version != AppState.get_version()
        ):
            self._run_profile()

    # ------------------------------------------------------------------ #
    #  UI                                                                  #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 32, 40, 32)
        layout.setSpacing(20)

        # Header row
        header_row = QHBoxLayout()
        title = QLabel("Explore")
        title.setObjectName("page_title")
        subtitle = QLabel("Column-level statistics and data quality overview.")
        subtitle.setObjectName("page_subtitle")

        self._refresh_btn = QPushButton("⟳  Refresh Profile")
        self._refresh_btn.setObjectName("secondary_btn")
        self._refresh_btn.clicked.connect(self._run_profile)

        header_row.addWidget(title)
        header_row.addStretch()
        header_row.addWidget(self._refresh_btn)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        # Top-level dataset summary bar
        self._summary_frame = QFrame()
        self._summary_frame.setObjectName("summary_bar")
        self._summary_frame.setVisible(False)
        summary_layout = QHBoxLayout(self._summary_frame)
        summary_layout.setContentsMargins(16, 8, 16, 8)

        self._summary_labels = {}
        for key in ["Rows", "Columns", "Numeric", "Categorical", "Missing cells"]:
            lbl = QLabel(f"{key}: —")
            lbl.setObjectName("summary_item")
            self._summary_labels[key] = lbl
            summary_layout.addWidget(lbl)
            if key != "Missing cells":
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.VLine)
                sep.setObjectName("summary_sep")
                summary_layout.addWidget(sep)

        layout.addWidget(self._summary_frame)

        # Status / error label
        self._status = QLabel("Import a dataset to see its profile.")
        self._status.setObjectName("status_label")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        # Scrollable column cards
        scroll = QScrollArea()
        scroll.setObjectName("cards_scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._scroll_content = QWidget()
        self._cards_grid = QGridLayout(self._scroll_content)
        self._cards_grid.setSpacing(16)
        self._cards_grid.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll.setWidget(self._scroll_content)
        layout.addWidget(scroll, 1)

    # ------------------------------------------------------------------ #
    #  Profiling                                                           #
    # ------------------------------------------------------------------ #

    def _run_profile(self):
        if self._worker is not None and self._worker.isRunning():
            return

        if AppState.get_dataframe() is None:
            self._on_error("No dataset loaded. Please import data first.")
            return

        self._status.setText("Profiling dataset…")
        self._summary_frame.setVisible(False)
        self._clear_cards()
        self._worker = ProfileWorker(AppState.get_version())
        self._worker.finished.connect(self._on_profile_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _clear_cards(self):
        while self._cards_grid.count():
            item = self._cards_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_profile_done(self, profile: dict, dataset_version: int):
        if dataset_version != AppState.get_version():
            self._worker = None
            self._run_profile()
            return

        self._status.setText("")
        self._profiled_version = dataset_version
        self._update_summary(profile)

        columns = profile.get("columns", {})
        for i, (col_name, stats) in enumerate(columns.items()):
            card = ColumnCard(col_name, stats)
            row, col = divmod(i, 3)
            self._cards_grid.addWidget(card, row, col)

        self._worker = None

    def _update_summary(self, profile: dict):
        meta = profile.get("meta", {})
        self._summary_labels["Rows"].setText(f"Rows: {meta.get('rows', '—'):,}")
        self._summary_labels["Columns"].setText(f"Columns: {meta.get('cols', '—')}")
        self._summary_labels["Numeric"].setText(f"Numeric: {meta.get('numeric_cols', '—')}")
        self._summary_labels["Categorical"].setText(f"Categorical: {meta.get('cat_cols', '—')}")
        self._summary_labels["Missing cells"].setText(f"Missing cells: {meta.get('total_missing', '—'):,}")
        self._summary_frame.setVisible(True)

    def _on_error(self, msg: str):
        self._status.setText(f"❌  {msg}")
        self._worker = None
