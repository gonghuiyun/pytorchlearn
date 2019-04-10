import numpy as np
import torch
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
from PIL import Image
import  matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def read_im():
    f = Image.open('digit.jpg').convert('L').resize((28,28))
    return f

def data_tf(x):
    x = np.array(x,dtype=np.float32)/255
    x = (x-0.5)/0.5
    x = x.reshape(-1)
    x = torch.from_numpy(x)
    return x

def label_final(x):
    x = x.numpy()
    index = int(x.argmax())
    return index


train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)
train_data = DataLoader(train_set,batch_size=32,shuffle = False)
train_data = iter(train_data)

seq = nn.Sequential(nn.Linear(784,3),
                    nn.ReLU(),
                    nn.Linear(3,4),
                    nn.ReLU(),
                    nn.Linear(4,10)
                    )



for layer in seq:
    if isinstance(layer,nn.Linear):
        w_layer_shape = layer.weight.shape
        init = (6**.5)/((w_layer_shape[0]+w_layer_shape[1])**.5)
        layer.weight.data = torch.from_numpy(np.random.uniform(-init,init,size=w_layer_shape)).float()

param = seq.parameters()
optim = torch.optim.SGD(param,lr = 1e-2,momentum = 0.9)
criterion = nn.CrossEntropyLoss()

for i in range(1000):
    data,y_label = next(train_data)
    print(data.shape)
    # data = Variable(data)
    # y_label = Variable(y_label)
    # y_out = seq(data)
    # # y_out = Variable(torch.argmax(y_out,dim = 1))
    # loss = criterion(y_out,y_label)
    #
    # optim.zero_grad()
    # loss.backward()
    # optim.step()
    #
    # print(loss)
    #
    # if i == 999:
    #     im_test = read_im()
    #     im_data = data_tf(im_test)
    #     y_test = seq(Variable(im_data))
    #     print(torch.argmax(y_test.data))



