def inf_ass(path):
    from Model import ShuffleNet
    from association import association
    import train_validation_inference as tvi
    model = ShuffleNet.ShuffleNetG3
    pred, value = tvi.inference(model, img_path=path)
    va = input("你的输入为：")
    if va == "":
        va = 1
    ans = value[int(va) - 1]
    print("你选择了：", ans)
    association.find(value[int(va) - 1])
    return ans


if __name__ == "__main__":
    import sys
    from IO import MyHccrWindow
    from PyQt5.QtWidgets import QApplication
    from IO.MyHccrWindow import MyHccrWindow

    # 手写板保存
    app = QApplication(sys.argv)
    MY = MyHccrWindow(inf_ass)
    MY.show()
    app.exec_()
