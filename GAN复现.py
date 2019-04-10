import torch
from torch import nn
from torch.autograd import Variable

import torchvision.transforms as tfs
from torch.utils.data import DataLoader, sampler
from torchvision.datasets import MNIST
from torchvision.utils import save_image

import numpy as np

def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5) / 0.5

# def deprocess_img(x):
#     return (x + 1.0) / 2.0

def to_img(x):
    x = (x + 1.0) / 2.0
    # x = x.clamp(0,1)
    x = x.view(x.shape[0],1,28,28)
    return x

class ChunkSampler(sampler.Sampler): # 定义一个取样的函数
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

NUM_TRAIN = 50000
NUM_VAL = 5000

NOISE_DIM = 96
batch_size = 128

train_set = MNIST('./data', train=True,  transform=preprocess_img)

train_data = DataLoader(train_set, batch_size=batch_size, sampler=ChunkSampler(NUM_TRAIN, 0))

# val_set = MNIST('./mnist', train=True, download=True, transform=preprocess_img)
#
# val_data = DataLoader(val_set, batch_size=batch_size, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

def discriminator():
    net = nn.Sequential(
            nn.Linear(784 , 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1)
    )
    return net

def genetor(noise_dim =NOISE_DIM):
    net = nn.Sequential(
            nn.Linear(noise_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,784),
            nn.Tanh()
    )
    return net

bce_loss =nn.BCEWithLogitsLoss()

def discriminator_loss(logits_real,logits_fake):
    size = logits_real.shape[0]
    true_labels =Variable(torch.ones(size,1)).float().cuda()
    false_labels = Variable(torch.zeros(size,1)).float().cuda()
    loss = bce_loss(logits_real,true_labels)+bce_loss(logits_fake,false_labels)
    return loss

def generator_loss(logits_fake):
    size = logits_fake.shape[0]
    true_labels = Variable(torch.ones(size,1)).float().cuda()
    loss = bce_loss(logits_fake,true_labels)
    return loss

def optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(),lr = 3e-4,betas = (0.5,0.999))
    return optimizer

def train_a_gan(D_net,G_net,D_optimizer,G_optimizer,discriminator_loss,generator_loss,noise_size = 96):
    for e in range(100):
        for x,_ in train_data:
            bs = x.shape[0]
            #判别网络
            real_data = Variable(x).view(bs,-1).cuda()
            logits_real = D_net(real_data)

            sample_noise = (torch.rand(bs,noise_size)-0.5)/0.5
            g_fake_seed = Variable(sample_noise).cuda()
            fake_images = G_net(g_fake_seed)
            logits_fake = D_net(fake_images)

            d_total_error = discriminator_loss(logits_real,logits_fake)
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step()

            #生成网络
            g_fake_seed = Variable(sample_noise).cuda()
            fake_images = G_net(g_fake_seed)

            gen_logits_fake = D_net(fake_images)
            g_error = generator_loss(gen_logits_fake)
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step()

        if (e+1)%20 ==0:
            print('d loss:{} ,g loss:{} '.format(d_total_error.data.item(),g_error.data.item()))
            save = to_img(fake_images.cpu().data)
            save_image(save, './picture/image_{}.png'.format(e+1))


D = discriminator().cuda()
G = genetor().cuda()

D_optim = optimizer(D)
G_optim = optimizer(G)

train_a_gan(D,G,D_optim,G_optim,discriminator_loss,generator_loss)