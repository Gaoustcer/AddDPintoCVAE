import torch.nn as nn
import torch
import torch.nn.functional as F
from math import sqrt
from math import log
def backwardhook(grad_in:torch.Tensor,C = 0.8,delta = 1e-3,epsilon = 0.01):
    z = (sqrt(2 * log(1.25/delta)))/epsilon
    grad_in.clip(-C,C)
    epsilon_grad = grad_in + z * C * torch.randn_like(grad_in)
    return epsilon_grad
    # pass
class Encoder(nn.Module):
    def __init__(self,add_noise = False,latentspacedim = 2) -> None:
        super(Encoder,self).__init__()
        self.flatten = nn.Flatten()
        self.feature = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(28**2 + 10,128),
            nn.ReLU(),
            nn.Linear(128,64)
        )
        self.muencode = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,latentspacedim)
        )
        from copy import deepcopy
        self.sigmaencode = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,latentspacedim)
            # nn.ReLU()
        )
        if add_noise:
            for param in self.parameters():
                param.register_hook(backwardhook)
    
    def forward(self,images,labels):
        images = self.flatten(images)
        feature = torch.concat([images,labels],-1)
        feature = self.feature(feature)
        return self.muencode(feature),self.sigmaencode(feature)

class Decoder(nn.Module):
    def __init__(self,add_noise=True,latentspacedim = 2) -> None:
        super(Decoder,self).__init__()
        self.Decode = nn.Sequential(
            nn.Linear(latentspacedim + 10,128),
            nn.ReLU(),
            nn.Linear(128,28**2),
            nn.Sigmoid()
        )
        if add_noise:
            for param in self.parameters():
                param.register_hook(backwardhook)
        
    
    def forward(self,sigma,mu,labels):
        noise = torch.randn_like(sigma).cuda()
        sigma = torch.exp(0.5 * sigma)
        
        # output result is log sigma
        feature = noise * sigma + mu
        feature = torch.concat([feature,labels],-1)
        return self.Decode(feature),feature


if __name__ == "__main__":
    net = Encoder(
    )
    net.parameters