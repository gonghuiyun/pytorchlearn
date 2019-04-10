import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import  DataLoader

from torchvision.datasets import MNIST

from torchvision import transforms as tfs
from torchvision.utils import save_image

im_tfs = tfs.Compose([tfs.ToTensor(),
                      tfs.Normalize([0.5],[0.5])
                      ])


train_set = MNIST('./data',transform=im_tfs)
train_data = DataLoader(train_set,batch_size=128,shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()

        self.encoder = nn.Sequential(nn.Linear(28*28,128),
                                     nn.ReLU(),
                                     nn.Linear(128,64),
                                     nn.ReLU(),
                                     nn.Linear(64,12),
                                     nn.ReLU(),
                                     nn.Linear(12,3)
                                    )

        self.decoder = nn.Sequential(nn.Linear(3,12),
                                     nn.ReLU(),
                                     nn.Linear(12,64),
                                     nn.ReLU(),
                                     nn.Linear(64,128),
                                     nn.ReLU(),
                                     nn.Linear(128,28*28),
                                     nn.Tanh()
                                    )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder,decoder


net = autoencoder()

criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(net.parameters(),lr = 1e-3)

def to_img(x):
    x = (x+0.5)*0.5
    x =x.clamp(0,1)
    x = x.view(x.shape[0],1,28,28)
    return x

i = 0
for data,_ in train_data:
    data = data.view(data.shape[0],-1)
    data = Variable(data)
    _,out = net(data)

    optimizer.zero_grad()
    loss = criterion(out,data)/data.shape[0]
    loss.backward()
    optimizer.step()
    i+=1
    if i%10 ==0:
        pic = to_img(out)
        save_image(pic, './picture/test{}.png'.format(i))

