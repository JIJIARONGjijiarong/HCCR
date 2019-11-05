import cv2
import numpy as np
def binaryzation(img:np.ndarray):
    """
    二值化
    :param img: 待处理的图像
    :return: 变换完成的二值化图像
    """
    imgs=cv2.imread('img')
    gray=cv2.cvtColor(imgs,cv2.COLOR_BGR2GRAY)#转换为灰度图像
    retval,dst=cv2.threshold(gray,0,255,cv2.THRESH_OTSU)#大津法实现二值化
    dst=cv2.dilate(dst,None,iterations=1)#膨胀
    dst=cv2.erode(dst,None,iterations=4)#腐蚀
    return dst
