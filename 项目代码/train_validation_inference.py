import argparse  # 提取命令行参数
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

parse = argparse.ArgumentParser(description='Params for training. ')
# 数据集根目录
parse.add_argument('--root', type=str, default='H:/', help='path to data set')
# 模式，3选1
parse.add_argument('--mode', type=str, default='train', choices=['train', 'validation', 'inference'])
# checkpoint 路径
parse.add_argument('--log_path', type=str, default='H:/log.pth', help='dir of check  points')

parse.add_argument('--restore', type=bool, default=True, help='whether to restore checkpoints')

parse.add_argument('--batch_size', type=int, default=32, help='size of mini-batch')
parse.add_argument('--image_size', type=int, default=32, help='resize image')
parse.add_argument('--epoch', type=int, default=100)
# 我的数据集类别数是3755，所以给定了一个选择范围
parse.add_argument('--num_class', type=int, default=3755, choices=range(10, 3755))
args = parse.parse_args()


def classes_txt(root, out_path, num_class=None):
    '''
    write image paths (containing class name) into a txt file.
    :param root: data set path
    :param out_path: txt file path
    :param num_class: how many classes needed
    :return: None
    '''
    dirs = os.listdir(root)  # 列出根目录下所有类别所在文件夹名
    if not num_class:  # 不指定类别数量就读取所有
        num_class = len(dirs)

    if not os.path.exists(out_path):  # 输出文件路径不存在就新建
        f = open(out_path, 'w')
        f.close()
    # 如果文件中本来就有一部分内容，只需要补充剩余部分
    # 如果文件中数据的类别数比需要的多就跳过
    with open(out_path, 'r+') as f:
        try:
            end = int(f.readlines()[-1].split('/')[-2]) + 1
        except:
            end = 0
        if end < num_class - 1:
            dirs.sort()
            dirs = dirs[end:num_class]
            for dir in dirs:
                files = os.listdir(os.path.join(root, dir))
                for file in files:
                    f.write(os.path.join(root, dir, file) + '\n')


class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, transforms=None):
        super(MyDataset, self).__init__()
        images = []  # 存储图片路径
        labels = []  # 存储类别名，在本例中是数字
        # 打开上一步生成的txt文件
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split('\\')[-2]) >= num_class:  # 只读取前 num_class 个类
                    break
                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split('\\')[-2]))
        self.images = images
        self.labels = labels
        self.transforms = transforms  # 图片需要进行的变换，ToTensor()等等

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')  # 用PIL.Image读取图像
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image)  # 进行变换
        return image, label

    def __len__(self):
        return len(self.labels)


def train(model):
    start = int(time.time())
    # 由于我的数据集图片尺寸不一，因此要进行resize，这里还可以加入数据增强，灰度变换，随机剪切等等
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    # 选择使用的设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = model()
    model.to(device)
    # 训练模式
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    # 由命令行参数决定是否从之前的checkpoint开始训练
    if args.restore:
        checkpoint = torch.load(args.log_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']
    else:
        loss = 0.0
        epoch = 0

    while epoch < args.epoch:
        train_set = MyDataset(args.root + "/data_" + str(epoch % 10) + '/train.txt', num_class=args.num_class,
                              transforms=transform)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            # 这里取出的数据就是 __getitem__() 返回的数据
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            end = int(time.time())
            print('epoch: %5d, batch: %5d, loss: %f, time: %5d' % (epoch + 1, i + 1, running_loss, end - start))
            running_loss = 0.0
            # 保存 checkpoint

        print('Save checkpoint...')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                   args.log_path)
        epoch += 1

    print('Finish training')


def validation(model):
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    test_set = MyDataset(args.root + "/data_19/train.txt", num_class=args.num_class, transforms=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model()
    model.to(device)

    checkpoint = torch.load(args.log_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += sum((predict == labels)).item()

            print('batch: %5d,\t acc: %f' % (i + 1, correct / total))
    print('Accuracy: %.2f%%' % (correct / total * 100))


def inference(model, img_path):
    def get_keys(d, value):
        return [k for k, v in d.items() if v == value]

    f = open("E:\\Git\\HCCR\\data\\char_dict.txt", "rb")
    dic = pickle.load(f)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    input = Image.open(img_path).convert('RGB')
    input = transform(input)
    input = input.unsqueeze(0)
    model = model()

    checkpoint = torch.load(args.log_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    output = model(input)
    _, pred = output.sort(1, descending=True)
    pred = list(pred.squeeze().data.numpy())[:10]
    value = []
    for i in range(10):
        value.append(get_keys(dic, pred[i])[0])
    print(value)
    return pred, value


if __name__ == '__main__':
    from Model import ShuffleNet

    model = ShuffleNet.ShuffleNetG3
    # classes_txt(args.root + '/data_11/train', args.root + 'data_11/train.txt', num_class=args.num_class)
    # classes_txt(args.root + '/test', args.root + '/test.txt', num_class=args.num_class)

    if args.mode == 'train':
        train(model)
    elif args.mode == 'validation':
        validation(model)
    elif args.mode == 'inference':
        inference(model, img_path='')
