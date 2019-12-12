import sys
from PyQt5.QtWidgets import QApplication
from IO.MyHccrWindow import MyHccrWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myhccr = MyHccrWindow()
    myhccr.show()
    app.exec_()
