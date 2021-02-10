import sys
import PySide2
from qtpy import QtWidgets


QApplication = QtWidgets.QApplication
APP = QApplication.instance()
if APP is None:
    APP = QApplication(sys.argv)


def main():
    launcher_window = QtWidgets.QMainWindow()
    launcher_window.setWindowTitle('Example Launcher')

    launcher_window.show()
    APP.exec_()


if __name__ == '__main__':
    main()
