"""Next Best Action page — stub (to be built in a later sprint)."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


class NextBestActionPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 32, 40, 32)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Next Best Action")
        title.setObjectName("page_title")

        subtitle = QLabel(
            "Coming soon — ML-driven next best action recommendations.\n"
            "Complete segmentation setup first."
        )
        subtitle.setObjectName("page_subtitle")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)
