import torch.nn as nn
import torch

# import 

class imagefeature(nn.Module):
    def __init__(self) -> None:
        super(imagefeature,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=2,stride=2),
            nn.Flatten()
        )
        self.encoderdim = 36
    
    def forward(self,images):
        return self.encoder(images)


class classifierbranch(nn.Module):
    def __init__(self) -> None:
        super(classifierbranch,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(36,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,10)
        )
    
    def forward(self,features):
        return self.classifier(features)

class reconstructbranch(nn.Module):
    def __init__(self) -> None:
        super(reconstructbranch,self).__init__()
        self.reconstruct = nn.Sequential(
            nn.Linear(36,128),
            nn.ReLU(),
            nn.Linear(128,28**2),
            nn.Sigmoid()
        )


    def forward(self,feature):
        return torch.reshape(self.reconstruct(feature),(-1,1,28,28))

class iddata(nn.Module):
    def __init__(self) -> None:
        super(iddata,self).__init__()
        self.imageencoder = imagefeature()
        self.classification = classifierbranch()
        self.reconstruction = reconstructbranch()
    def forward(self,images):
        feature = self.imageencoder(images)
        return feature,self.classification(feature),self.reconstruction(feature)


if __name__ == "__main__":
    images = torch.rand(3,1,28,28)
    iddetection = iddata()
    element = iddetection(images)
    for e in element:
        print(e.shape)
    # featureencoder = imagefeature()
    # feature = featureencoder(images)
    # print("feature",feature.shape)
    # recon = reconstructbranch()
    # reconresult = recon(feature)
    # print("reconstruction result ",reconresult.shape)