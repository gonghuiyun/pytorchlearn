import numpy
import torch
from torch import nn
from torch.nn import init

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.l1 = nn.Sequential( nn.Linear(40,30),
                                 nn.ReLU())
        init.xavier_normal_(self.l1[0].weight)
        self.l2 = nn.Sequential(nn.Linear(30,20),
                                nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(20,10),
                                nn.ReLU())
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2(x)

a = net()
# for i in a.children():
#     print(i)
print(a.parameters())