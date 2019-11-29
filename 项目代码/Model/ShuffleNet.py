import argparse  # 提取命令行参数
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset

parse = argparse.ArgumentParser(description='')
# 数据集根目录
parse.add_argument('--root', type=str, default='E:\Git\HCCR\data', help='path to data set')
# 模式，3选1
parse.add_argument('--mode', type=str, default='train', choices=['train', 'validataion', 'inference'])
# checkpoint 路径
parse.add_argument('--log_path', type=str, default=os.path.abspath('.') + 'E:\Git\HCCR\data\log\ShuffleNet.pth', help='dir of checkpoints')

parse.add_argument('--restore', type=bool, default=False, help='whether to restore checkpoints')

parse.add_argument('--batch_size', type=int, default=32, help='size of mini-batch')

parse.add_argument('--image_size', type=int, default=32, help='resize image')

parse.add_argument('--epoch', type=int, default=100)

parse.add_argument('--num_class', type=int, default=3755, choices=range(10, 3755))

args = parse.parse_args()


def classes_txt(root, out_path, num_class=None):
    """
    write image paths (containing class name) into a txt file.
    :param root: dataSet path
    :param out_path: txt file path
    :param num_class: classes number
    :return: None
    """
    dirs = os.listdir(root)
    if not num_class:
        num_class = len(dirs)

    if not os.path.exists(out_path):
        f = open(out_path, 'w')
        f.close()

    with open(out_path, 'r+') as f:
        try:
            end = int(f.readlines()[-1].split(os.sep)[-2]) + 1
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
        images = []
        labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split(os.sep)[-2]) >= num_class:
                    break
                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split(os.sep)[-2]))
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)


import time


def train(overal_situation_writer):
    start = int(time.time())
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    train_set = MyDataset(args.root + '/train.txt', num_class=args.num_class, transforms=transform)
    # print('data size:', train_set.__len__())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = ShuffleNetG3()

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        train_writer = SummaryWriter(comment=str(epoch))
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            running_loss = 0.0
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            end = int(time.time())
            train_writer.add_scalar("loss", loss, global_step=i, walltime=end - start)
        #     sys.stdout.write(
        #         "time %5ds,epoch: %5d,batch: %5d,loss: %f  \r" % (end - start, epoch, i, running_loss))
        #     sys.stdout.flush()
        # print('Save checkpoint:%5d,loss: %f,used time %5d' % (epoch, running_loss, end - start))
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                   args.log_path)
        validation(overal_situation_writer, train_writer)
        epoch += 1
        overal_situation_writer.add_scalar("epoch-loss",running_loss,global_step=epoch)
        train_writer.close()
    print('Finish training')


# 将测试图片送入模型进行测试
def validation(overal_situation_writer, writer):
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    test_set = MyDataset(args.root + '/test.txt', num_class=args.num_class, transforms=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ShuffleNetG3()
    model.to(device)

    checkpoint = torch.load(args.log_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += sum(predict == labels).item()
            writer.add_scalar("Accuracy", correct / total, global_step=i)
    #         if i % 10 == 0:
    #             sys.stdout.write(
    #                 "batch: %5d,acc: %f                         \r" % (i, correct / total))
    #             sys.stdout.flush()
    # print('Accuracy: %.2f%%' % (correct / total * 100))
    overal_situation_writer.add_scalar("Accuracy",(correct / total * 100))


def inference(img_path):
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Grayscale(),
        transforms.ToTensor()])

    input = Image.open(img_path).convert('RGB')
    input = transform(input)
    input = input.unsqueeze(0)
    model = ShuffleNetG2()
    output = model(input)
    _, pred = torch.max(output.data, 1)

    print('predict:\t%4d' % pred)


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        # 维度变换之后必须要使用.contiguous()使得张量在内存连续之后才能调用view函数
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        # bottleneck层中间层的channel数变为输出channel数的1/4
        mid_planes = int(out_planes / 4)

        g = 1 if in_planes == 24 else groups
        # 作者提到不在stage2的第一个pointwise层使用组卷积,因为输入channel数量太少,只有24
        self.conv1 = nn.Conv2d(in_planes, mid_planes,
                               kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes,
                               kernel_size=3, stride=stride, padding=1,
                               groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes,
                               kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, cfg):
        super(ShuffleNet, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.conv1 = nn.Conv2d(1, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], 3755)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(Bottleneck(self.in_planes,
                                         out_planes - self.in_planes,
                                         stride=2, groups=groups))
            else:
                layers.append(Bottleneck(self.in_planes,
                                         out_planes,
                                         stride=1, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ShuffleNetG2():
    cfg = {
        'out_planes': [200, 400, 800],
        'num_blocks': [4, 8, 4],
        'groups': 2
    }
    return ShuffleNet(cfg)


def ShuffleNetG3():
    cfg = {
        'out_planes': [240, 480, 960],
        'num_blocks': [4, 8, 4],
        'groups': 3
    }
    return ShuffleNet(cfg)


if __name__ == '__main__':
    overal_situation_writer = SummaryWriter(comment="overal_situation")
    classes_txt(args.root + '/train', args.root + '/train.txt', num_class=args.num_class)
    classes_txt(args.root + '/test', args.root + '/tets.txt', num_class=args.num_class)
    if args.mode == 'train':
        train(overal_situation_writer)
    elif args.mode == 'validation':
        validation(None,None)
    elif args.mode == 'inference':
        inference('')
