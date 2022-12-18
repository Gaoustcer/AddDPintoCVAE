import torch
import torch.nn as nn
from torch.utils.data import Dataset

class membershipclassification(nn.Module):
    def __init__(self) -> None:
        super(membershipclassification,self).__init__()
        self.infer = nn.Sequential(
            nn.Linear(10,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,2)
        )
    
    def forward(self,classifcationfeature):
        return self.infer(classifcationfeature)

from Membershipinfer.Membershipdataset import Membershipdataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
class Membershipinfer(object):
    def __init__(self,targetmodel = "./logs/models/targetmodel",traindatasetpath = "./logs/traindataset",testdatasetpath = "./logs/testdataset",EPOCH = 32) -> None:
        membershipdataset = Membershipdataset(traindatasetpath=traindatasetpath,testdatasetpath=testdatasetpath)
        # trainnumber = int(len(membershipdataset//2))
        self.validatedataset = torch.load(traindatasetpath)
        self.validateloader = DataLoader(self.validatedataset,batch_size=32)
        self.validatelen = len(self.validatedataset)
        self.traindataloader = DataLoader(membershipdataset,batch_size=32)
        # trainmembershipinfer,testmembershipinfer = random_split(membershipdataset,(trainnumber,trainnumber))
        # self.trainloader = DataLoader(,batch_size=32)
        # self.testloader = DataLoader(testmembershipinfer,batch_size=32)
        self.net = membershipclassification().cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr = 0.001)
        self.EPOCH = EPOCH

    def validate(self):
        same = 0
        for images,_ in self.testloader:
            images = images.cuda()
            # labels = labels.cuda()
            predlabels = self.net(images)
            # same +=  sum(predlabels == 1)
            same + torch.sum(torch.max(predlabels).indices)
        return same/self.validatelen
    
    def train(self):
        for epoch in range(self.EPOCH):
            from tqdm import tqdm
            for images,labels in tqdm(self.traindataloader):
                self.optimizer.zero_grad()
                predlabels = self.net(images)
                loss = F.cross_entropy(predlabels,labels)
                loss.backward()
                self.optimizer.step()
            accrate = self.validate()
            print("epoch {}".format(epoch),"acc rate {}".format(accrate))
                



# if __name__ ==/
    