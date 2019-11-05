"""
Python_IO模块
包括：
    图像读取
    图像显示
    图像保存
"""
import cv2

import numpy as np

def load_img(file_path: str):
    """
    图像读取函数
    :param file_path: 文件路径
    :return: 读取的图像
    """
    img = cv2.imread(file_path)
    return img


def show_img(wname: str, img: np.ndarray):
    """
    图像显示函数(按q退出)
    :param wname: 图像的显示名称
    :param img: 待显示的图像
    :return: None
    """
    cv2.imshow(wname, img)
    if cv2.waitKey() == ord("q"):
        cv2.destroyWindow(wname)


def save_img(filename: str, img: np.ndarray):
    """
    图像保存函数
    :param filename: 图像的路径/名称.格式
    :param img: 待保存的图像
    :return: None
    """
    cv2.imwrite(filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
