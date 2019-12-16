import argparse
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

parse = argparse.ArgumentParser(description='Params for training. ')
parse.add_argument('--root', type=str, default='E:\Git\HCCR\data', help='path to data set')
parse.add_argument('--mode', type=str, default='train', choices=['train', 'validation', 'inference'])
parse.add_argument('--log_path', type=str, default='vgg_log.pth', help='dir of checkpoints')
parse.add_argument('--restore', type=bool, default=False, help='whether to restore checkpoints')
parse.add_argument('--batch_size', type=int, default=16, help='size of mini-batch')
parse.add_argument('--image_size', type=int, default=224, help='resize image')
parse.add_argument('--epoch', type=int, default=10)
parse.add_argument('--num_class', type=int, default=100, choices=range(10, 3755))
args = parse.parse_args()


class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, transforms=None):
        super(MyDataset, self).__init__()
        images = []
        labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split('\\')[-2]) >= num_class:  # just get images of the first #num_class
                    break
                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split('\\')[-2]))
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


def train():
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    # transforms.Grayscale(),
                                    transforms.ToTensor()])

    train_set = MyDataset(args.root + '/train.txt', num_class=args.num_class, transforms=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = VGG(make_layers(cfg['E'], batch_norm=True), num_classes=args.num_class)
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
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 200 == 199:  # every 200 steps
                print('epoch %5d: batch: %5d, loss: %f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

        if epoch % 10 == 9:
            print('Save checkpoint...')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss},
                       args.log_path)
        epoch += 1

    print('Finish training')


def validation():
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    test_set = MyDataset(args.root + '/test.txt', num_class=args.num_class, transforms=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VGG(make_layers(cfg['E'], batch_norm=True), num_classes=args.num_class)
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
            correct += sum(int(predict == labels)).item()

            if i % 100 == 99:
                print('batch: %5d,\t acc: %f' % (i + 1, correct / total))
    print('Accuracy: %.2f%%' % (correct / total * 100))


def inference():
    print('Start inference...')
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    f = open(args.root + '/test.txt')
    num_line = sum(line.count('\n') for line in f)
    f.seek(0, 0)
    line = int(torch.rand(1).data * num_line - 10)  # -10 for '\n's are more than lines
    while line > 0:
        f.readline()
        line -= 1
    img_path = f.readline().rstrip('\n')
    f.close()
    label = int(img_path.split('/')[-2])
    print('label:\t%4d' % label)
    input = Image.open(img_path).convert('RGB')
    input = transform(input)
    input = input.unsqueeze(0)
    model = VGG(make_layers(cfg['E'], batch_norm=True), num_classes=args.num_class)
    model.eval()
    checkpoint = torch.load(args.log_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    output = model(input)
    _, pred = torch.max(output.data, 1)

    print('predict:\t%4d' % pred)


def classes_txt(root, out_path, num_class=None):
    '''
    write image paths (containing class name) into a txt file.
    :param root: data set path
    :param out_path: txt file path
    :param num_class: how many classes needed
    :return: None
    '''
    dirs = os.listdir(root)
    if not num_class:
        num_class = len(dirs)

    if not os.path.exists(out_path):
        f = open(out_path, 'w')
        f.close()

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


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model


if __name__ == '__main__':

    classes_txt(args.root + '/train', args.root + '/train.txt', num_class=args.num_class)
    classes_txt(args.root + '/test', args.root + '/test.txt', num_class=args.num_class)

    if args.mode == 'train':
        train()
    elif args.mode == 'validation':
        validation()
    elif args.mode == 'inference':
        inference()
