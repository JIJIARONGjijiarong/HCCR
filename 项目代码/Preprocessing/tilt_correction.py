import numpy as np
import cv2
import math
from scipy import ndimage

def edge_detection(img: np.ndarray):
    """
    边缘检测
    :param img: 待变换的灰度图像
    :return: 变换完成的灰度图像
    """
    return cv2.Canny(img, 50, 150, apertureSize=3)

def tilt_correction(img: np.ndarray):
    """
    倾斜校正
    :param img: 待变换的灰度图像
    :return: 变换完成的灰度图像
    """
    edge = edge_detection(img)  # 图像的边缘检测
    lines = cv2.HoughLines(edge, 1, np.pi / 180, 0)
    x1, x2, y1, y2 = 0, 0, 0, 0
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
    if x1 == x2 or y1 == y2:
        return img
    t = float(y2 - y1) / (x2 - x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    rotate_img = ndimage.rotate(img.copy(), rotate_angle, cval=255)
    return rotate_img