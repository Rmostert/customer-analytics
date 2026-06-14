"""
Import Data page — CSV, Excel, JSON, Parquet.

Threading rules enforced here:
  - LoadWorker runs I/O in a background thread (no UI calls, no AppState writes).
  - _on_loaded() runs on the main thread via Qt signal; it is the ONLY place
    AppState is written to.
  - _worker reference is kept so we never start two loads simultaneously.
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QFrame, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QComboBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from app.core.data_loader import DataLoader, LoadResult
from app.utils.app_state import AppState


class LoadWorker(QThread):
    """
    Background thread — pure I/O, zero Qt or AppState interaction.
    Emits finished(LoadResult) on success, error(str) on failure.
    """
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, filepath: str, encoding: str = "utf-8"):
        super().__init__()
        self.filepath = filepath
        self.encoding = encoding

    def run(self):
        try:
            self.progress.emit(20)
            result = DataLoader.load(self.filepath, encoding=self.encoding)
            self.progress.emit(90)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class DropZone(QFrame):
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("drop_zone")
        self.setAcceptDrops(True)
        self.setFixedHeight(160)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("📂")
        icon.setObjectName("drop_icon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        text = QLabel("Drag & drop your file here")
        text.setObjectName("drop_text")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtext = QLabel("Supports CSV, Excel (.xlsx), JSON, Parquet")
        subtext.setObjectName("drop_subtext")
        subtext.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(icon)
        layout.addWidget(text)
        layout.addWidget(subtext)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("dragover", True)
            self.style().polish(self)

    def dragLeaveEvent(self, event):
        self.setProperty("dragover", False)
        self.style().polish(self)

    def dropEvent(self, event: QDropEvent):
        self.setProperty("dragover", False)
        self.style().polish(self)
        urls = event.mimeData().urls()
        if urls:
            self.file_dropped.emit(urls[0].toLocalFile())


class ImportPage(QWidget):
    def __init__(self):
        super().__init__()
        self._worker: LoadWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 32, 40, 32)
        layout.setSpacing(20)

        title = QLabel("Import Data")
        title.setObjectName("page_title")

        subtitle = QLabel(
            "Load your customer dataset. "
            "Supported formats: CSV, Excel, JSON, Parquet. "
            "CSV and Parquet files over 500 MB are queried via DuckDB without loading into memory."
        )
        subtitle.setObjectName("page_subtitle")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        # Drop zone
        self._drop_zone = DropZone()
        self._drop_zone.file_dropped.connect(self._load_file)
        layout.addWidget(self._drop_zone)

        # Browse row
        browse_row = QHBoxLayout()
        browse_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        or_label = QLabel("— or —")
        or_label.setObjectName("or_label")
        self._browse_btn = QPushButton("Browse Files…")
        self._browse_btn.setObjectName("primary_btn")
        self._browse_btn.setFixedWidth(160)
        self._browse_btn.clicked.connect(self._browse_file)
        browse_row.addWidget(or_label)
        browse_row.addWidget(self._browse_btn)
        layout.addLayout(browse_row)

        # Encoding selector
        enc_row = QHBoxLayout()
        enc_label = QLabel("CSV encoding:")
        enc_label.setObjectName("field_label")
        self._enc_combo = QComboBox()
        self._enc_combo.addItems(["utf-8", "latin-1", "cp1252", "utf-16"])
        self._enc_combo.setFixedWidth(120)
        enc_row.addStretch()
        enc_row.addWidget(enc_label)
        enc_row.addWidget(self._enc_combo)
        layout.addLayout(enc_row)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setObjectName("load_progress")
        self._progress.setVisible(False)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        # Status row
        status_row = QHBoxLayout()
        status_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status = QLabel("")
        self._status.setObjectName("status_label")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._engine_badge = QLabel("")
        self._engine_badge.setObjectName("engine_badge")
        self._engine_badge.setVisible(False)
        status_row.addWidget(self._status)
        status_row.addWidget(self._engine_badge)
        layout.addLayout(status_row)

        # Stat cards
        self._stats_frame = QFrame()
        self._stats_frame.setObjectName("stats_row")
        self._stats_frame.setVisible(False)
        stats_layout = QHBoxLayout(self._stats_frame)
        stats_layout.setContentsMargins(0, 0, 0, 0)

        self._stat_rows  = self._make_stat_card("Rows",      "—")
        self._stat_cols  = self._make_stat_card("Columns",   "—")
        self._stat_size  = self._make_stat_card("File size", "—")
        self._stat_nulls = self._make_stat_card("Missing",   "—")

        for card in [self._stat_rows, self._stat_cols,
                     self._stat_size, self._stat_nulls]:
            stats_layout.addWidget(card)

        layout.addWidget(self._stats_frame)

        # Preview table
        self._preview_label = QLabel("Preview (first 100 rows)")
        self._preview_label.setObjectName("section_label")
        self._preview_label.setVisible(False)

        self._table = QTableWidget()
        self._table.setObjectName("preview_table")
        self._table.setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        layout.addWidget(self._preview_label)
        layout.addWidget(self._table)

    def _make_stat_card(self, label: str, value: str) -> QFrame:
        card = QFrame()
        card.setObjectName("stat_card")
        vbox = QVBoxLayout(card)
        vbox.setSpacing(2)
        val_lbl = QLabel(value)
        val_lbl.setObjectName("stat_value")
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        key_lbl = QLabel(label)
        key_lbl.setObjectName("stat_key")
        key_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(val_lbl)
        vbox.addWidget(key_lbl)
        card._value_label = val_lbl
        return card

    # ------------------------------------------------------------------ #
    #  File loading                                                        #
    # ------------------------------------------------------------------ #

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Dataset", "",
            "Data Files (*.csv *.xlsx *.xls *.json *.parquet);;All Files (*)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        # Guard: ignore if a load is already in progress
        if self._worker is not None and self._worker.isRunning():
            return

        if not os.path.exists(path):
            self._show_error(f"File not found: {path}")
            return

        self._engine_badge.setVisible(False)
        self._status.setText(f"Loading {os.path.basename(path)}…")
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._browse_btn.setEnabled(False)

        # Build worker, wire ALL signals, THEN start — order matters
        self._worker = LoadWorker(path, self._enc_combo.currentText())
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_loaded)
        self._worker.error.connect(self._show_error)
        self._worker.start()                         # start last

    # ------------------------------------------------------------------ #
    #  Slot — runs on main thread via Qt signal delivery                  #
    # ------------------------------------------------------------------ #

    def _on_loaded(self, result: LoadResult):
        self._progress.setValue(100)
        self._progress.setVisible(False)
        self._browse_btn.setEnabled(True)

        # Write to AppState here (main thread), never inside the worker
        if result.is_large:
            AppState.set_duckdb(result.duckdb_con, result.filepath)
            AppState.set_preview_df(result.df)
            AppState.set_meta(
                result.row_count,
                result.col_count,
                result.column_names,
                result.column_types,
                load_encoding=result.load_encoding,
            )
        else:
            AppState.set_dataframe(
                result.df, result.filepath, load_encoding=result.load_encoding,
            )

        self._status.setText(
            f"✅  Loaded — {result.row_count:,} rows × {result.col_count} columns"
        )

        if result.is_large:
            self._engine_badge.setText("⚡ DuckDB mode")
            self._engine_badge.setVisible(True)

        self._populate_preview(result.df)
        self._populate_stats(result)

    def _populate_preview(self, df):
        if df is None:
            return
        self._table.setRowCount(100)
        self._table.setColumnCount(len(df.columns))
        self._table.setHorizontalHeaderLabels(list(df.columns))
        for r, (_, row) in enumerate(df.head(100).iterrows()):
            for c, val in enumerate(row):
                self._table.setItem(r, c, QTableWidgetItem(str(val)))
        self._table.setVisible(True)
        self._preview_label.setVisible(True)

    def _populate_stats(self, result: LoadResult):
        file_size = os.path.getsize(result.filepath)
        if file_size >= 1024 ** 3:
            size_str = f"{file_size / 1024 ** 3:.2f} GB"
        elif file_size >= 1024 ** 2:
            size_str = f"{file_size / 1024 ** 2:.1f} MB"
        else:
            size_str = f"{file_size / 1024:.1f} KB"

        self._stat_rows._value_label.setText(f"{result.row_count:,}")
        self._stat_cols._value_label.setText(str(result.col_count))
        self._stat_size._value_label.setText(size_str)

        if result.is_large:
            self._stat_nulls._value_label.setText("see Explore")
        elif result.df is not None:
            self._stat_nulls._value_label.setText(
                f"{int(result.df.isnull().sum().sum()):,}"
            )
        self._stats_frame.setVisible(True)

    def _show_error(self, msg: str):
        self._progress.setVisible(False)
        self._browse_btn.setEnabled(True)
        self._status.setText(f"❌  {msg}")
        QMessageBox.critical(self, "Import Error", msg)
