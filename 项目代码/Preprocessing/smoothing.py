import cv2
import numpy as np


def mean_filtering(img: np.ndarray, size=(3, 3)):
    """
    均值滤波
    :param img: 待降噪的图像
    :param size: 滤波器大小
    :return: 降噪完成的图像
    """
    me = cv2.blur(img, size)
    return me


def median_filtering(img: np.ndarray, size=3):
    """
    中值滤波
    :param img: 待降噪的图像
    :param size: 滤波器的尺寸
    :return: 降噪完成后的图像
    """
    med = cv2.medianBlur(img, size)
    return med
