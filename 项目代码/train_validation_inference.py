import argparse  # 提取命令行参数
import os
import pickle
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset

parse = argparse.ArgumentParser(description='')

parse.add_argument('--root', type=str, default='E:\\Git\\HCCR\\data', help='path to data set')

parse.add_argument('--mode', type=str, default='train', choices=['train', 'validataion', 'inference'])

parse.add_argument('--log_path', type=str, default=os.path.abspath('.') + "E:\\Git\HCCR\\data\\log\\ShuffleNet.pth",
                   help="dir of checkpoints")

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


def train(model):
    start = int(time.time())
    transform = transforms.Compose([transforms.RandomRotation(degrees=30),
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    train_set = MyDataset(args.root + '/train.txt', num_class=args.num_class, transforms=transform)
    print('data size:', train_set.__len__())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = model

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
            sys.stdout.write(
                "time %5ds,epoch: %5d,batch: %5d,loss: %f  \r" % (end - start, epoch, i, running_loss))
            sys.stdout.flush()
        print('Save checkpoint:%5d,loss: %f,used time %5d' % (epoch, running_loss, end - start))
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                   args.log_path)
        validation(model)
        epoch += 1
        writer.add_scalar("epoch-loss", running_loss, global_step=epoch)
        train_writer.close()
    print('Finish training')


def validation(model):
    transform = transforms.Compose([transforms.RandomRotation(degrees=30),
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    test_set = MyDataset(args.root + '/test.txt', num_class=args.num_class, transforms=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model
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
            if i % 10 == 0:
                sys.stdout.write(
                    "batch: %5d,acc: %f                         \r" % (i, correct / total))
                sys.stdout.flush()
    print('Accuracy: %.2f%%' % (correct / total * 100))
    writer.add_scalar("Accuracy", (correct / total * 100))


def inference(model, img_path):
    def get_keys(d, value):
        return [k for k, v in d.items() if v == value]

    dic = pickle.load('')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Grayscale(),
        transforms.ToTensor()])

    input = Image.open(img_path).convert('RGB')
    input = transform(input)
    input = input.unsqueeze(0)
    model = model
    model.to(device)

    checkpoint = torch.load(args.log_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    output = model(input)
    _, pred = torch.max(output.data, 1)

    value = get_keys(dic, pred)

    print('predict:\t%4d' % pred)
    print(value)


if __name__ == '__main__':
    from Model import ShuffleNet

    model = ShuffleNet.ShuffleNetG3()
    writer = SummaryWriter(comment="overal_situation")
    classes_txt(args.root + '/train', args.root + '/train.txt', num_class=args.num_class)
    classes_txt(args.root + '/test', args.root + '/tets.txt', num_class=args.num_class)
    if args.mode == 'train':
        train(model)
    elif args.mode == 'validation':
        validation(model)
    elif args.mode == 'inference':
        inference(model, '')
