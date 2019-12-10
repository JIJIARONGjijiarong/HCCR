import sys
from PyQt5.QtWidgets import (QApplication, QWidget)
from PyQt5.QtGui import (QPainter, QPen)
from PyQt5.QtCore import Qt

# import cv2

class Example(QWidget):
    def __init__(self):
        super(Example, self).__init__()

        # resize设置宽高，move设置位置
        self.resize(500, 500)
        self.move(100, 100)
        self.setWindowTitle("简单的画板4.0")

        # setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        # self.setMouseTracking(False)

        '''
        要想将按住鼠标后移动的轨迹保留在窗体上
        需要一个列表来保存所有移动过的点
        '''
        self.pos_xy = []

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 15, Qt.SolidLine)
        painter.setPen(pen)

        '''
        首先判断pos_xy列表中是不是至少有两个点了
        然后将pos_xy中第一个点赋值给point_start
        利用中间变量pos_tmp遍历整个pos_xy列表
        point_end = pos_tmp

        判断point_end是否是断点，如果是
        point_start赋值为断点
        continue
        判断point_start是否是断点，如果是
        point_start赋值为point_end
        continue

        画point_start到point_end之间的线
        point_start = point_end
        这样，不断地将相邻两个点之间画线，就能留下鼠标移动轨迹了
        '''
        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent(self, event):
        '''
        按住鼠标移动事件：将当前点添加到pos_xy列表中
        调用update()函数在这里相当于调用paintEvent()函数
        每次update()时，之前调用的paintEvent()留下的痕迹都会清空
        '''
        # 中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        # pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        '''
        重写鼠标按住后松开的事件
        在每次松开后向pos_xy列表中添加一个断点(-1, -1)
        然后在绘画时判断一下是不是断点就行了
        是断点的话就跳过去，不与之前的连续
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        # self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pyqt_learn = Example()
    print(pyqt_learn)
    pyqt_learn.show()
    sys.exit(app.exec_())
    # cv2.destroyAllWindows()