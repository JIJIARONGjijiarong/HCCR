import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets,transforms

writer = SummaryWriter()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5))
])
