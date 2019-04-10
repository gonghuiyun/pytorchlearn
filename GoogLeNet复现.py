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

def conv_relu(in_channel,out_channel,kernel,stride = 1,padding = 0):
    layer = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel,stride,padding),
                        nn.BatchNorm2d(out_channel,eps = 1e-3),
                         nn.ReLU()
                        )
    return layer

class inception(nn.Module):
    def __init__(self,in_channel,out1_1,out2_1,out2_3,out3_1,out3_5,out4_1):
        super(inception,self).__init__()
        self.path1 = conv_relu(in_channel,out1_1,1)
        self.path2 = nn.Sequential(conv_relu(in_channel,out2_1,1),
                                    conv_relu(out2_1,out2_3,3,padding = 1)
                                   )
        self.path3 = nn.Sequential(conv_relu(in_channel,out3_1,1),
                                    conv_relu(out3_1,out3_5,5,padding=2)
                                   )
        self.path4 = nn.Sequential( nn.MaxPool2d(3,stride=1,padding=1),
                                    conv_relu(in_channel,out4_1,1)
                                    )
    def forward(self,x):
        f1 = self.path1(x)
        f2= self.path2(x)
        f3 = self.path3(x)
        f4 = self.path4(x)
        output = torch.cat((f1,f2,f3,f4),dim = 1)
        return output

# net = inception(3,64,48,64,64,96,32)
# test = Variable(torch.zeros(1,3,96,96))
# print(test.shape)
# out = net(test)
# print(out.shape)

class GoogLeNet(nn.Module):
    def __init__(self, in_channel, num_classes,verbose):
        super(GoogLeNet,self).__init__()
        self.verbose = verbose
        self.block1 = nn.Sequential(conv_relu(in_channel,64,kernel= 7,stride=2,padding=3),
                                    nn.MaxPool2d(3,2)
                                    )
        self.block2 = nn.Sequential(conv_relu(64,64,kernel=1),
                                    conv_relu(64,192,kernel=3,padding=1),
                                    nn.MaxPool2d(3,2)
                                    )
        self.block3 = nn.Sequential(inception(192,64,96,128,16,32,32),
                                    inception(256,128,128,192,32,96,64),
                                    nn.MaxPool2d(3, 2)
                                    )
        self.block4 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )
        self.block5 = nn.Sequential(inception(832, 256, 160, 320, 32, 128, 128),
                                    inception(832, 384, 182, 384, 48, 128, 128),
                                    nn.AvgPool2d(2)
                                    )
        self.classifier = nn.Linear(1024,num_classes)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
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

net = GoogLeNet(3,10,False)
train(train_data,net)


