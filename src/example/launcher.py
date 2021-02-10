import logging
import sys
import subprocess
import os
import PySide2

from qtpy import QtWidgets
from qtpy import QtCore
from qtpy import QtGui


LOGGER = logging.getLogger(__name__)

try:
    QApplication = QtGui.QApplication
except AttributeError:
    QApplication = QtWidgets.QApplication

APP = QApplication.instance()
if APP is None:
    APP = QApplication(sys.argv)  # pragma: no cover



def main():
    launcher_window = QtWidgets.QMainWindow()
    launcher_window.setWindowTitle('InVEST Launcher')

    launcher_window.show()
    APP.exec_()


if __name__ == '__main__':
    main()
