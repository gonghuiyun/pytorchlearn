import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torch import nn
from torch.autograd import Variable
# from utils import train

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x

train_set = CIFAR10('./data', train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
# train_data =iter(train_data)
test_set = CIFAR10('./data', train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
# test_data = iter(test_data)

def vgg_block(num_convs, in_channels,out_channels):
    net = [nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),nn.ReLU()]
    for i in range(num_convs-1):
        net.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        net.append(nn.ReLU())
    net.append(nn.MaxPool2d(2,2))
    return nn.Sequential(*net)

def vgg_stack(num_convs,channels):
    net = []
    for i, j in zip(num_convs,channels):
        net.append(vgg_block(i,j[0],j[1]))
    return nn.Sequential(*net)

vgg_feature = vgg_stack((1,1,2,2,2),((3,64),(64,128),(128,256),(256,512),(512,512)))

class vgg(nn.Module):
    def __init__(self):
        super(vgg,self).__init__()
        self.feature = vgg_feature
        self.fc1 = nn.Sequential(nn.Linear(512,100),
                                 nn.ReLU(),
                                 nn.Linear(100,10)
                                )
    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        return x

def train(train_data,model):
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    # optimizer = torch.optim.Adam(model.parameters(),lr = 1e-2,betas = (0.9,0.99))
    criterion = nn.CrossEntropyLoss()
    for epoch in range(20):
        model = model.train()
        loss = 0
        for train, label in train_data:
            if torch.cuda.is_available():
                train = Variable(train.cuda())
                label = Variable(label.cuda())
            else:
                train = Variable(train)
                label = Variable(label)

            y_out = model(train)
            loss = criterion(y_out , label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(loss)

net = vgg()
train(train_data,net)






# model = vgg_block(3,64,128)
# input = Variable(torch.zeros(3,64,300,300))
# print(input.data.shape)
# out = model(input)
# print(model)
# print(out.shape)
# print(vgg_feature)