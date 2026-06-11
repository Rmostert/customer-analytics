"""Segmentation page — stub (to be built in next sprint)."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


class SegmentationPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 32, 40, 32)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Segmentation")
        title.setObjectName("page_title")

        subtitle = QLabel(
            "Coming next — K-Means clustering and RFM segmentation.\n"
            "Import and explore your data first."
        )
        subtitle.setObjectName("page_subtitle")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)
