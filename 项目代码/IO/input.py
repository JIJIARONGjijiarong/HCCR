"""
input模块
包括：
    单幅图像的读取
    整个目录下图像的读取
    整个数据集的读取
"""
import os
import cv2
import numpy as np


def load_img(file_path: str):
    """
    图像读取函数
    :param file_path: 文件路径
    :return: 读取的图像
    """
    img = cv2.imread(file_path)
    img = cv2.resize(img, (32, 32), cv2.INTER_LINEAR)
    return img


def load_dir(dir_path: str, data: list, target):
    """
    目录读取函数
    :param dir_path: 目录路径
    :param data: 数据
    :param target: 类别
    """
    file_list = os.listdir(dir_path)
    os.chdir(dir_path)
    tar = dir_path.split(os.altsep)[-1]

    for img in file_list:
        data.append(load_img(img))
        target.append(tar)


def load_data(data_path):
    """
    数据读取函数
    :param data_path: 数据路径
    :return: 读取的数据，图像的类别
    """
    data_dir = os.listdir(data_path)
    os.chdir(data_path)
    data = []
    target = []
    for dir_path in data_dir:
        load_dir(dir_path, data, target)
        os.chdir(data_path)

    return np.array(data), np.array(target)


if __name__ == "__main__":
    dataSet, targets = load_data("H:/图片")
