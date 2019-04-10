import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
#                     [9.779], [6.182], [7.59], [2.167], [7.042],
#                     [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
#
# y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
#                     [3.366], [2.596], [2.53], [1.221], [2.827],
#                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

def poly(x):
    return torch.mm(x,w)+b

def loss_func(y_out,y):
    return torch.mean((y_out - y)**2)

w = np.array([0.5,3,2.4])
b = np.array([0.9])

x_sample = np.arange(-3, 3.1, 0.1)
y_sample = w[0]*x_sample + w[1]*x_sample**2 +w[2]*x_sample**3 +b
x_connect = np.stack([x_sample**i for i in range(1,4)], axis=1)

w = Variable(torch.randn(3, 1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)

x_connect = Variable(torch.from_numpy(x_connect).float(),requires_grad = True)
y_train = Variable(torch.from_numpy(y_sample).float(),requires_grad = True)

for i in range(10):
    y_out = poly(x_connect)
    plt.plot(x_sample, y_sample, label='origin')
    y_display = y_out.detach()
    print(y_train.type())
    plt.plot(x_sample, y_display.data.numpy(), label='train')
    plt.legend()
    plt.show()

    loss = loss_func(y_out,y_train)
    loss.backward()

    w.data = w.data - 1e-2*w.grad.data
    b.data = b.data - 1e-2*b.grad.data









