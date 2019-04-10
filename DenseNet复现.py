
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torch import nn
from torch.autograd import Variable
# from utils import train

def data_tf(x):
    x = x.resize((96, 96), 2)
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x

train_set = CIFAR10('./data', train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = CIFAR10('./data', train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False)
    )
    return layer


class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate

        self.net = nn.Sequential(*block)

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

def transition(in_channel,out_channel):
    layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                          nn.ReLU(),
                          nn.Conv2d(in_channel,out_channel,1),
                          nn.AvgPool2d(2,2))
    return layer

class densenet(nn.Module):
    def __init__(self,in_channel,growth_rate = 32,block_layers = [6,12,24,16]):
        super(densenet,self).__init__()
        # self.block1 = nn.Sequential(nn.Conv2d(in_channel,64,7,2,3),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(3,2,padding = 1))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
        )

        block = []
        channels = 64
        for i,layers in enumerate(block_layers):
            block.append(dense_block(channels,growth_rate,layers))
            channels += layers*growth_rate
            if i<len(block_layers)-1:
                block.append(transition(channels,channels//2))
                channels = channels//2
        self.block2 = nn.Sequential(*block)
        self.block2.add_module('bn',nn.BatchNorm2d(channels))
        self.block2.add_module('relu',nn.ReLU())
        self.block2.add_module('avgpool', nn.AvgPool2d(3))
        self.classifier = nn.Linear(channels,10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)
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

net = densenet(3)
train(train_data,net)





