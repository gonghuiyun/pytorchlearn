import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image

im_tfs =tfs.Compose([tfs.ToTensor(),
                     tfs.Normalize([0.5],[0.5])
                    ])
train_set =MNIST('./data',transform= im_tfs)
train_data =DataLoader(train_set,batch_size=128,shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        self.f1 = nn.Linear(28*28,400)
        self.f21 = nn.Linear(400 ,20)
        self.f22 = nn.Linear(400 ,20)
        self.f3 = nn.Linear(20,400)
        self.f4 = nn.Linear(400 ,28*28)

    def encoder(self,x):
        x = F.relu(self.f1(x))
        return self.f21(x),self.f22(x)

    def reparametrize(self,mu,logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        if torch.cuda.is_available():
            eps = Variable(eps.cuda())
        else:
            eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decoder(self,z):

        z = F.relu(self.f3(z))
        z = torch.tanh(self.f4(z))
        return z

    def forward(self, x):
        mu,logvar = self.encoder(x)
        z = self.reparametrize(mu,logvar)
        out = self.decoder(z)
        return out,mu,logvar

def to_img(x):
    x = x*0.5+0.5
    x = x.clamp(0,1)
    x = x.view(x.shape[0],1,28,28)
    return x

reconstruction_function = nn.MSELoss(size_average=False)
def loss_function(recon_x,x,mu,logvar):
    MSE = reconstruction_function(recon_x,x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return MSE+KLD

net = VAE()
if torch.cuda.is_available():
    net = net.cuda()

x,lable = train_set[0]
x = x.view(x.shape[0],-1)
if torch.cuda.is_available():
    x = x.cuda()
x =Variable(x)
out,mu,logvar = net(x)
print(mu)

optimizer = torch.optim.Adam(net.parameters(),lr = 1e-3)
i = 0
for e in range(100):
    for im,label in train_data:
        im = im.view(im.shape[0],-1)
        im = Variable(im)
        if torch.cuda.is_available():
            im = im.cuda()
        recon_im ,mu,logvar = net(im)
        optimizer.zero_grad()
        loss = loss_function(recon_im,im,mu,logvar)/im.shape[0]
        loss.backward()
        optimizer.step()
        if (e+1) % 20 == 0:
            print(loss)
            save = to_img(recon_im.cpu().data)
            save_image(save, './picture/image_{}.png'.format(e+1))
        i+=1


