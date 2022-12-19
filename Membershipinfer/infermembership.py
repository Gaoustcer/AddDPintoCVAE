import torch
import torch.nn as nn
from torch.utils.data import Dataset

def initlayer(layer):
    if isinstance(layer,nn.Linear):
        nn.init.uniform_(layer.weight,a = -0.1,b = 0.1)
        nn.init.constant_(layer.bias, 0.1)

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
        # for m in self.modules():
            
            # print(m)
            # m.weight.data.normal_(0,1)
            # m.bias.data.zero_()
    
    def forward(self,classifcationfeature):
        return self.infer(classifcationfeature)

from Membershipinfer.Membershipdataset import Membershipdataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
class Membershipinfer(object):
    def __init__(self,targetmodel = "./logs/models/targetmodel",traindatasetpath = "./logs/traindataset",testdatasetpath = "./logs/testdataset",EPOCH = 32) -> None:
        membershipdataset = Membershipdataset(traindatasetpath=traindatasetpath,testdatasetpath=testdatasetpath)
        # trainnumber = int(len(membershipdataset//2))
        self.validatedataset = torch.load(traindatasetpath)
        self.validateloader = DataLoader(self.validatedataset,batch_size=32,shuffle=True)
        self.validatelen = len(self.validatedataset)
        self.traindataloader = DataLoader(membershipdataset,batch_size=32,shuffle=True)
        # trainmembershipinfer,testmembershipinfer = random_split(membershipdataset,(trainnumber,trainnumber))
        # self.trainloader = DataLoader(,batch_size=32)
        # self.testloader = DataLoader(testmembershipinfer,batch_size=32)
        self.net = membershipclassification().cuda()
        self.net.apply(initlayer)
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr = 0.001)
        self.targetmodel = torch.load(targetmodel).cuda()
        self.EPOCH = EPOCH
        self.writer = SummaryWriter("./logs/privacyattack")
        
        # self.writer = 

    def validate(self):
        same = 0
        for images,_ in self.validateloader:
            with torch.no_grad():
                images = images.cuda()
                feature = self.targetmodel(images)
                # print("feature is",feature.shape)
                # labels = labels.cuda()
                predlabels = self.net(feature)
                # same +=  sum(predlabels == 1)
                # print(predlabels.shape)
                # print(predlabels)
                same += torch.sum(torch.max(predlabels,dim=-1).indices,dim=-1)
        print("same number",same)
        return same/self.validatelen
    
    def train(self):
        index = 0
        for epoch in range(self.EPOCH):
            from tqdm import tqdm
            accrate = self.validate()
            print("epoch {}".format(epoch),"acc rate {}".format(accrate))
            for images,labels in tqdm(self.traindataloader):
                self.optimizer.zero_grad()
                
                predlabels = self.net(self.targetmodel(images).detach())
                # print(predlabels)
                # print("pred labels and labels are",predlabels.shape,labels.shape)
                loss = F.cross_entropy(predlabels,labels)
                # print("train label is",labels)
                loss.backward()
                self.writer.add_scalar("loss",loss,index)
                index += 1
                self.optimizer.step()
            
                



# if __name__ ==/
    