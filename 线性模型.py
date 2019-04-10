import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x_train = np.array([[3.3],[4.4]],dtype= np.float32)
y_train = np.array([[1.7],[2.76]],dtype= np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

x_train = Variable(x_train)
y_train = Variable(y_train)

w = Variable(torch.rand(1),requires_grad= True)
b = Variable(torch.zeros(1),requires_grad= True)

def linear_model(x):
    return w*x + b

def get_loss(y_out,y):
    return torch.mean((y_out-y)**2)

for i in range(1):
    y_out = linear_model(x_train)
    print(y_out)

    plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
    plt.plot(x_train.data.numpy(), y_out.data.numpy(), 'ro', label='estimate')
    plt.legend()
    plt.show()

    loss = get_loss(y_out,y_train)
    loss.backward()

    w.data = w.data - 1e-2 * w.grad.data
    b.data = b.data - 1e-2 * b.grad.data





