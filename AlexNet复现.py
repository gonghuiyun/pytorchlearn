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

# train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)
# train_data = DataLoader(train_set,batch_size=32,shuffle = False)
# train_data = iter(train_data)

train_set = CIFAR10('./data', train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
train_data =iter(train_data)
test_set = CIFAR10('./data', train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
test_data = iter(test_data)

def aac(test_data,model):
    count = 0
    for i in range(len(test_data)):
        test , label_real = next(test_data)
        test = Variable(test)
        label_real = Variable(label_real)
        label_pre = model(test)
        list = np.argmax(label_pre.data.numpy(),axis = 1)
        print(torch.from_numpy(list))
        print(label_real)

        # print(label_real.shape)
    #     if label_real.data == label_pre.data:
    #         count +=1
    # return count/len(test_data)

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.l1 = nn.Sequential(nn.Conv2d(3,64,5),
                           nn.ReLU()
                           )
        self.l2 = nn.MaxPool2d(3,2)
        self.l3 = nn.Sequential(nn.Conv2d(64,64,5,1),
                           nn.ReLU()
                           )
        self.l4 = nn.MaxPool2d(3, 2)
        self.l5 = nn.Sequential(nn.Linear(1024,384),
                           nn.ReLU()
                           )
        self.l6 = nn.Sequential(nn.Linear(384,192),
                           nn.ReLU()
                           )
        self.l7 = nn.Sequential(nn.Linear(192,10),
                           # nn.ReLU()
                           )


    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        x = x.view(x.shape[0],-1)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return x

def train(train_data,model):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    # optimizer = torch.optim.Adam(model.parameters(),lr = 1e-2,betas = (0.9,0.99))
    criterion = nn.CrossEntropyLoss()

    for i in range(len(train_data)):
        train , label = next(train_data)
        train = Variable(train)
        label = Variable(label)

        y_out = model(train)
        loss = criterion(y_out , label)
        if i%100 == 0:
            print(loss)
            print(model.l1[0].weight.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model = net()
train(train_data,model)

