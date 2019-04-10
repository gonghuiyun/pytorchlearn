import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

def logic(x,w,b):
    return torch.sigmoid(torch.mm(x,w)+b)

def loss_func(y_out,y):
    loss = (y*y_out.clamp(1e-12).log()+(1-y)*(1-y_out).clamp(1e-12).log()).mean()
    return -loss

with open('data.txt','r') as f:
    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]),float(i[1]),float(i[2])) for i in data_list]

    x_max =  max([i[0] for i in data])
    y_max = max([i[1] for i in data])
    data = [(i[0]/x_max,i[1]/y_max,i[2])for i in data]

    index_0 = list(filter(lambda x: x[-1]==0,data))
    index_1 = list(filter(lambda x: x[-1] == 1, data))

    x_0 = [i[0] for i in index_0]
    y_0 = [i[1] for i in index_0]

    x_1 = [i[0] for i in index_1]
    y_1 = [i[1] for i in index_1]




data1 = np.array(data,dtype=np.float32)
x = data1[:,0:2]
index = data1[:,2:3]


data1 = Variable(torch.from_numpy(x),requires_grad = True)
index_real = Variable(torch.from_numpy(index),requires_grad = True)

# w = Variable(torch.randn(2,1).float(),requires_grad = True)
# b = Variable(torch.zeros(1).float(),requires_grad = True)
w = nn.Parameter(torch.randn(2,1).float())
b = nn.Parameter(torch.zeros(1).float())
optimizer = torch.optim.SGD([w,b],lr=1.)
criterion = nn.BCEWithLogitsLoss()

for i in range(1000):
    index_out = logic(data1,w,b)
    # loss = loss_func(index_out,index_real)
    loss = criterion(index_out,index_real)

    optimizer.zero_grad()
    loss.backward()

    # w.data = w.data - 1e-1*w.grad.data
    # b.data = b.data - 1e-1*b.grad.data
    #
    # w.grad.zero_()
    # b.grad.zero_()

    optimizer.step()

    if i%100 == 0:
        print(loss)
        # if loss <0.59:
        w0 = w.data[0].numpy()
        w1 = w.data[1].numpy()
        b0 = b.data.numpy()
        x_plot = np.arange(0.2, 1, 0.01)
        y_plot = (-w0 * x_plot - b0) / w1
        plt.plot(x_0, y_0, 'ro', label='index 0')
        plt.plot(x_1, y_1, 'bo', label='index 1')
        plt.plot(x_plot, y_plot, 'g', label='logic')
        plt.legend()
        plt.show()




