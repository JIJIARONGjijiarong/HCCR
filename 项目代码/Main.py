import train_validation_inference as tvi

if __name__ == "__main__":
    from Model import ShuffleNet
    from IO import MyHccrWindow
    from association import associat
    import sys
    from PyQt5.QtWidgets import QApplication
    from IO.MyHccrWindow import MyHccrWindow

    # 导入手写板


    # 手写板保存
    app = QApplication(sys.argv)
    model = ShuffleNet.ShuffleNetG3
    MY = MyHccrWindow(model,tvi.inference)
    MY.show()
    app.exec_()
    # pred, value = tvi.inference(model, img_path=path)
    # va = input("你的输入为：")
    # if va == "":
    #     va = 1
    # print("你选择了：",value[int(va)-1])
    # associat.associat(value[int(va)-1])