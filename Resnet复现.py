import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torch import nn
import torch.nn.functional as F
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

def conv3x3(in_channel,out_channel,kernel = 3, stride = 1,padding = 1):
    return nn.Conv2d(in_channel,out_channel,kernel_size= kernel,stride=stride,padding = padding)

class residual_block(nn.Module):
    def __init__(self,in_channel,out_channel,same_shape = True):
        super(residual_block,self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = conv3x3(in_channel,out_channel,stride =stride)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = conv3x3(out_channel,out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel,out_channel,1,stride = stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out),True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out),True)
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out+x,True)

class resnet(nn.Module):
    def __init__(self,in_channel,num_classes,verbose):
        super(resnet,self).__init__()
        self.verbose = verbose
        self.num_classes = num_classes

        self.block1 = nn.Conv2d(in_channel,64,7,2)

        self.block2 = nn.Sequential(
                                    nn.MaxPool2d(2,2),
                                    residual_block(64,64),
                                    residual_block(64, 64),
                                    )
        self.block3 = nn.Sequential(

                                    residual_block(64,128,False),
                                    residual_block(128, 128),
                                    )
        self.block4 = nn.Sequential(

                                    residual_block(128,256,False),
                                    residual_block(256, 256),
                                    )
        self.block5 = nn.Sequential(

                                    residual_block(256,512,False),
                                    residual_block(512, 512),
                                    nn.AvgPool2d(3)
                                    )
        self.classifier = nn.Linear(512,num_classes)

    def forward(self, x):
        out = self.block1(x)
        if self.verbose:
            print(out.shape)
        out = self.block2(out)
        if self.verbose:
            print(out.shape)
        out = self.block3(out)
        if self.verbose:
            print(out.shape)
        out = self.block4(out)
        if self.verbose:
            print(out.shape)
        out = self.block5(out)
        if self.verbose:
            print(out.shape)

        out = out.view(x.shape[0],-1)
        out = self.classifier(out)
        return out

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

net = resnet(3,10,False)
train(train_data,net)

