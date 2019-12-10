import train_validation_inference as tvi

if __name__ == "__main__":
    import time
    from Model import ShuffleNet
    from association import associat
    # 导入手写板
    # 手写板保存
    path = 'E:\Git\HCCR\data\\test\\00000\\3215.png'
    model = ShuffleNet.ShuffleNetG3()
    start = int(time.time())
    ans_num,ans = tvi.inference(model, img_path=path)
    end = int(time.time())
    print(end - start)
    associat.get_lianxiang1(ans[0])