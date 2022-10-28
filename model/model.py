import torch.nn as nn
import torch
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self) -> None:
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
            nn.Linear(32,16)
        )
        from copy import deepcopy
        self.sigmaencode = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16)
            # nn.ReLU()
        )
    
    def forward(self,images,labels):
        images = self.flatten(images)
        feature = torch.concat([images,labels],-1)
        feature = self.feature(feature)
        return self.muencode(feature),self.sigmaencode(feature)

class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder,self).__init__()
        self.Decode = nn.Sequential(
            nn.Linear(16 + 10,128),
            nn.ReLU(),
            nn.Linear(128,28**2),
            nn.Sigmoid()
        )
    
    def forward(self,sigma,mu,labels):
        noise = torch.randn_like(sigma).cuda()
        sigma = torch.exp(0.5 * sigma)
        
        # output result is log sigma
        feature = noise * sigma + mu
        feature = torch.concat([feature,labels],-1)
        return self.Decode(feature)


if __name__ == "__main__":
    net = Encoder(
    )
    net.parameters