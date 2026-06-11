"""
Customer Analytics Desktop App
Entry point - run this file to launch the application.
"""

import sys
import os

# Ensure the project root is on the path regardless of where Python is launched from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from app.ui.main_window import MainWindow


def main():
    # Enable high-DPI scaling
    app = QApplication(sys.argv)
    app.setApplicationName("Customer Analytics")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Your Company")

    # Load global stylesheet
    with open("assets/style.qss", "r") as f:
        app.setStyleSheet(f.read())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
