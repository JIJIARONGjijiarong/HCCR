import train_validation_inference as tvi

if __name__ == "__main__":
    import time
    from Model import ShuffleNet
    from association import associat
    # 导入手写板
    # 手写板保存
    path = 'E:\Git\HCCR\data\\test\\00004\\46213.png'
    model = ShuffleNet.ShuffleNetG3()
    start = int(time.time())
    pred, value = tvi.inference(model, img_path=path)
    end = int(time.time())
    associat.get_lianxiang1(value[0])