import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt

def readimg():
    img = Image.open('digit.jpg').convert('L')
    img = np.array(img,dtype=np.float32)
    img = img.reshape(1,1,img.shape[0],img.shape[1])
    img = torch.from_numpy(img)
    return img



conv1 = nn.Conv2d(1,1,3,bias = False)

sobel_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype = np.float32)
sobel_kernel = sobel_kernel.reshape(1,1,3,3)
conv1.weight.data = torch.from_numpy(sobel_kernel)

pool1 = nn.MaxPool2d(2,2)

out = conv1(readimg())
out = pool1(out)

print(out.data.squeeze().numpy())
plt.imshow(out.data.squeeze().numpy(),cmap = 'gray')
plt.show()
