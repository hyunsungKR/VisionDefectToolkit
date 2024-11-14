# main.py

import sys
from PyQt6.QtWidgets import QApplication
from controller import FilterController

if __name__ == '__main__':
    app = QApplication(sys.argv)
    controller = FilterController()
    sys.exit(app.exec())
