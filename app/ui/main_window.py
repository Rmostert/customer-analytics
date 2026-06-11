"""
Main application window - sets up the sidebar navigation and content area.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QFont

from app.ui.pages.import_page import ImportPage
from app.ui.pages.explore_page import ExplorePage
from app.ui.pages.segmentation_page import SegmentationPage
from app.ui.pages.nba_page import NextBestActionPage


NAV_ITEMS = [
    ("📂", "Import Data",       ImportPage),
    ("🔍", "Explore",           ExplorePage),
    ("🎯", "Segmentation",      SegmentationPage),
    ("⚡", "Next Best Action",  NextBestActionPage),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Customer Analytics")
        self.setMinimumSize(1200, 760)
        self._nav_buttons = []

        self._build_ui()
        self._navigate(0)  # Start on Import page

    # ------------------------------------------------------------------ #
    #  UI Construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_sidebar())
        layout.addWidget(self._build_content_area())

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)

        vbox = QVBoxLayout(sidebar)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Logo / brand area
        brand = QLabel("CA")
        brand.setObjectName("brand_logo")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.setFixedHeight(72)

        brand_label = QLabel("Customer\nAnalytics")
        brand_label.setObjectName("brand_label")
        brand_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_label.setFixedHeight(48)

        vbox.addWidget(brand)
        vbox.addWidget(brand_label)

        # Divider
        divider = QFrame()
        divider.setObjectName("sidebar_divider")
        divider.setFrameShape(QFrame.Shape.HLine)
        vbox.addWidget(divider)
        vbox.addSpacing(12)

        # Nav buttons
        for i, (icon, label, _) in enumerate(NAV_ITEMS):
            btn = QPushButton(f"  {icon}  {label}")
            btn.setObjectName("nav_btn")
            btn.setCheckable(True)
            btn.setFixedHeight(48)
            btn.clicked.connect(lambda checked, idx=i: self._navigate(idx))
            self._nav_buttons.append(btn)
            vbox.addWidget(btn)

        vbox.addStretch()

        # Version label
        version = QLabel("v1.0.0")
        version.setObjectName("version_label")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setFixedHeight(32)
        vbox.addWidget(version)

        return sidebar

    def _build_content_area(self):
        self._stack = QStackedWidget()
        self._stack.setObjectName("content_area")

        for _, _, PageClass in NAV_ITEMS:
            self._stack.addWidget(PageClass())

        return self._stack

    # ------------------------------------------------------------------ #
    #  Navigation                                                          #
    # ------------------------------------------------------------------ #

    def _navigate(self, index: int):
        self._stack.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == index)
