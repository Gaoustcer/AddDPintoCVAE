import torch.nn as nn
import torch
import torch.nn.functional as F
from math import sqrt
from math import log
def backwardhook(grad_in:torch.Tensor,C = 0.8,delta = 1e-5,epsilon = 10):
    z = (sqrt(2 * log(1.25/delta)))/epsilon
    # grad_in.clip(-C,C)
    normal = max(1,torch.norm(grad_in,p=2)/C)
    grad_in /= normal
    epsilon_grad = grad_in + z * C * torch.randn_like(grad_in)
    return epsilon_grad

'''
clip: g = g/(1,\parallel g\parallel/2)
add noise into grad N(0,\sigma^2I)
will gurantee (\epsilon,\delta)-DP
\sigma = \sqrt{2\log \frac{1.25}{\delta}} /\epsilon 
epsilon = 10,delta = 10^{-5}
\sigma = 
'''
    # pass
class Encoder(nn.Module):
    def __init__(self,add_noise = False,latentspacedim = 2) -> None:
        super(Encoder,self).__init__()
        self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(num_embeddings=10,
            embedding_dim=32
        )
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(36,32)
        )
        # self.feature = nn.Sequential(
        #     # nn.Flatten(),
        #     nn.Linear(28**2 + 10,128),
        #     nn.ReLU(),
        #     nn.Linear(128,32)
        # )
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
        # images = self.flatten(images)
        feature = self.feature(images)
        labels = self.embedding(labels)
        feature = torch.concat([feature,labels],-1)
        # feature = torch.concat([images,labels],-1)
        # print("feature shape is",feature.shape)
        # feature = self.feature(feature)
        return self.muencode(feature),self.sigmaencode(feature)

class Decoder(nn.Module):
    def __init__(self,add_noise=True,latentspacedim = 8) -> None:
        super(Decoder,self).__init__()
        # self.feature 
        # self.Decode = nn.Sequential(
        #     nn.Linear(latentspacedim + 32,128),
        #     nn.ReLU(),
        #     nn.Linear(128,28**2),
        #     nn.Sigmoid()
        # )
        self.Decode = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.ConvTranspose1d(in_channels=1,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1)
        )
        self.Imagegenerate = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1,out_channels=1,
                kernel_size=(3,4),
                stride=(2,3),
                padding=(3,2),
                output_padding=1),
            nn.Sigmoid()
        )
        self.embedding = nn.Embedding(num_embeddings=10,
            embedding_dim=32)
        self.latentdim = latentspacedim
        if add_noise:
            for param in self.parameters():
                param.register_hook(backwardhook)
        
    
    def forward(self,sigma,mu,labels):
        labels = self.embedding(labels)
        if sigma != None and mu != None:
            noise = torch.randn_like(sigma).cuda()
            sigma = torch.exp(0.5 * sigma)
            
            # output result is log sigma
            feature = noise * sigma + mu
        else:
            feature = torch.normal(0,1,(len(labels),self.latentdim)).cuda()
        # print("labels shape",labels.shape)
        # print('sigma shape',sigma.shape)
        # print("feature shape is",feature.shape)
        feature = torch.concat([feature,labels],-1).unsqueeze(1)
        # print(feature.shape)
        images = self.Decode(feature).reshape(-1,1,16,10)
        return self.Imagegenerate(images),feature
        # return self.Decode(feature),feature


if __name__ == "__main__":
    net = Encoder(
    )
    net.parameters