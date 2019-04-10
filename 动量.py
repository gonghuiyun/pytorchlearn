import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def data_tf(x):
    x = np.array(x , dtype=np.float32)/255
    x = (x-0.5)/0.5
    x = x.reshape(-1)
    x = torch.from_numpy(x).float()
    return x

train_set = MNIST('./data',train = True,transform=data_tf,download=False)

def grad_func(parameters,lr):
    for layer in parameters:
        layer.data = layer.data - lr * layer.grad.data

train_data = DataLoader(train_set,batch_size=64 ,shuffle = True)

net = nn.Sequential(nn.Linear(784,200),
                    nn.ReLU(),
                    nn.Linear(200,10),
                    )

criterion = nn.CrossEntropyLoss()
optimzier = torch.optim.Adam(net.parameters(),lr = 1e-2,betas = (0.9,0.999))


loss_final = []
idx = 0
for i in range(5):
    train_loss = 0
    for x_train , label in train_data:
        print(x_train.shape)
#         x_train = Variable(x_train)
#         label = Variable(label)
#         y_out = net(x_train)
#         loss = criterion(y_out,label)
#         # train_loss += loss
#         if idx % 300 == 0:
#             loss_final.append(loss.item())
#         optimzier.zero_grad()
#         loss.backward()
#         optimzier.step()
#         # grad_func(net.parameters(),1e-2)
#
#         print(loss)
#
# x_axis = np.linspace(0,5,len(loss_final),endpoint=True)
# plt.semilogy(x_axis,loss_final)
# plt.show()





