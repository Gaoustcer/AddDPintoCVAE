from model.smood import iddata as Indistribution
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from dataset import data_train
from torch.utils.data import DataLoader
import torch.nn.functional as F


class Indistributiontrain(object):
    def __init__(self,log_path ="./logs/OODdetection/IIDtrain") -> None:
        self.Indistribution = Indistribution().cuda()
        self.logpath = log_path
        self.modelpath = os.path.join(self.logpath,"modelsaved")
        if os.path.exists(self.modelpath) == False:
            os.mkdir(self.modelpath)
        self.modelpath = os.path.join(self.modelpath,"model")
        self.writer = SummaryWriter(self.logpath)
        self.dataset = data_train
        self.loader = DataLoader(self.dataset,batch_size=32)
        self.EPOCH = 16
        self.index = 0
        self.optim = torch.optim.Adam(self.Indistribution.parameters(),lr = 0.001)
    
    def train(self):
        for epoch in range(self.EPOCH):
            self.trainanepoch()
    
    def trainanepoch(self):
        from tqdm import tqdm
        for images,labels in tqdm(self.loader):
            self.optim.zero_grad()
            images = images.cuda()
            labels = labels.cuda()
            _,classification,reconstructionimages = self.Indistribution(images)
            classificationloss = F.cross_entropy(classification,labels)
            self.writer.add_scalar("classificationloss",classificationloss,self.index)
            reconstructloss = F.binary_cross_entropy(reconstructionimages,images)
            self.writer.add_scalar("reconstructloss",reconstructloss,self.index)
            self.index += 1
    
    def save(self):
        torch.save(self.Indistribution,self.modelpath)


if __name__ == "__main__":
    trainobject = Indistributiontrain()
    trainobject.train()