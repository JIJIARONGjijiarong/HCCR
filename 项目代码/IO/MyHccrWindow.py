# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 08:46:31 2019

@author: 邬洲
"""

#import tensorflow as tf
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel)
from PyQt5.QtGui import (QPainter, QPen, QFont)
from PyQt5.QtCore import Qt
from PIL import ImageGrab, Image

class MyHccrWindow(QWidget):

    def __init__(self,model,inference):
        super(MyHccrWindow, self,).__init__()
        self.model = model
        self.inference = inference
        self.resize(480, 500)  # resize设置宽高
        self.move(100, 100)    # move设置位置
        self.setWindowFlags(Qt.FramelessWindowHint)  # 窗体无边框
        #setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)

        self.pos_xy = []  #保存鼠标移动过的点

        # 添加一系列控件
        self.label_draw = QLabel('', self)
        self.label_draw.setGeometry(2, 2, 476, 450)
        self.label_draw.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_draw.setAlignment(Qt.AlignCenter)

        self.label_result_name = QLabel('结果：', self)
        self.label_result_name.setGeometry(20, 460, 60, 35)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel(' ', self)
        self.label_result.setGeometry(84, 460, 35, 35)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_result.setAlignment(Qt.AlignCenter)

        self.btn_recognize = QPushButton("识别", self)
        self.btn_recognize.setGeometry(130,460, 50, 35)
        self.btn_recognize.clicked.connect(self.btn_recognize_on_clicked)

        self.btn_clear = QPushButton("清空", self)
        self.btn_clear.setGeometry(190, 460, 50, 35)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)

        self.btn_close = QPushButton("关闭", self)
        self.btn_close.setGeometry(250, 460, 50, 35)
        self.btn_close.clicked.connect(self.btn_close_on_clicked)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 15, Qt.SolidLine)
        painter.setPen(pen)

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
        '''
        #中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        #pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        '''
            重写鼠标按住后松开的事件
            在每次松开后向pos_xy列表中添加一个断点(-1, -1)
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

    def btn_recognize_on_clicked(self):
        bbox = (104, 104, 572,550)
        im = ImageGrab.grab(bbox)    # 截屏，手写数字部分
        im = im.resize((32, 32), Image.ANTIALIAS)  # 将截图转换成 32 * 32 像素
        im.save("screenshot.jpg")

        pred, value = self.inference(self.model,"screenshot.jpg")
        recognize_result = value[0]

        self.label_result.setText(str(recognize_result))  # 显示识别结果
        self.update()

    def btn_clear_on_clicked(self):
        self.pos_xy = []
        self.label_result.setText('')
        self.update()

    def btn_close_on_clicked(self):
        self.close()
