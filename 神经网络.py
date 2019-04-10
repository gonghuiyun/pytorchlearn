import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

np.random.seed(1)
m = 400 # 样本数量
N = int(m/2) # 每一类的点的个数
D = 2 # 维度
x = np.zeros((m, D))
y = np.zeros((m, 1), dtype='uint8') # label 向量，0 表示红色，1 表示蓝色
a = 4

for j in range(2):
    ix = range(N*j,N*(j+1))
    t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
    r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)
plt.show()

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

seq_net=nn.Sequential(
        nn.Linear(2,4),
        nn.Tanh(),
        nn.Linear(4,1)
)

param = seq_net.parameters()
criterion = nn.BCEWithLogitsLoss()
optim = torch.optim.SGD(param,1.)



for i in range(1000):
    y_out = seq_net(Variable(x))
    loss = criterion(y_out,Variable(y))
    print(loss)
    optim.zero_grad()
    loss.backward()
    optim.step()

    if loss<0.29:
        print(seq_net(Variable(torch.from_numpy(np.array([2])))))
