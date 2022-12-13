from model.smood import iddata as Indistribution
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from dataset import data_train
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import data_test
from generatedataset import mixturedataset
from torch.utils.data import random_split
from model.smood import classifierood

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
            loss = classificationloss + reconstructloss
            loss.backward()
            self.optim.step()
            self.index += 1
    
    def save(self):
        torch.save(self.Indistribution,self.modelpath)


class trainforsmoothclassifier(object):
    # from dataset import data_test
    # from generatedataset import mixturedataset
    def __init__(self,load_pretrain_model = True,model_path = "./logs/IIDtrain/modelsaved/model",real_dataset = data_test,log_path = "./logs/OODdetection/OODdetect") -> None:
        if load_pretrain_model:
            self.model = torch.load(model_path).cuda()
        else:
            raise NotImplementedError
        dataset = mixturedataset(real_dataset)
        trainlen = 18000
        validationlen = len(self.dataset) - trainlen
        self.validationlen = validationlen
        traindataset, testdataset = random_split(dataset,(trainlen,validationlen))
        self.trainloader = DataLoader(traindataset,batch_size=32)
        self.testloader = DataLoader(testdataset,batch_size=32)
        self.index = 0
        self.EPOCH = 32
        self.logpath = log_path
        self.writer = SummaryWriter(self.logpath)
        self.modelpath = os.path.join(self.logpath,"model")
        self.classification = classifierood().cuda()
        self.optimizer = torch.optim.Adam()

    
    def train(self):
        from tqdm import tqdm
        for epoch in range(self.EPOCH):
            for images,reallabels in tqdm(self.trainloader):
                if self.index % 32 == 0:
                    accrate = self.validate()
                    self.writer.add_scalar("accrate",accrate,self.index//32)
                self.optimizer.zero_grad()
                images = images.cuda()
                reallabels = reallabels.cuda()
                with torch.no_grad():
                    features,labels,reconstructionresult = self.model(images)
                classification = self.classification(features,labels,reconstructionresult)
                loss = F.mse_loss(classification,reallabels)
                loss.backward()
                self.writer.add_scalar("loss",loss,self.index)
                self.index += 1
                self.optimizer.step()

                pass
    def validate(self):
        samecount = 0
        from tqdm import tqdm
        with torch.no_grad():
            for images,reallabels in tqdm(self.testloader):
                images = images.cuda()
                reallabels = reallabels.cuda()
                features,labels,reconstructionresult = self.model(images)
                classification = self.classification(features,labels,reconstructionresult)
                maxindices = torch.max(classification,dim=-1).indices
                samecount += sum(maxindices == reallabels)
        return samecount/self.validationlen

        # pass

        # pass
    # pass

if __name__ == "__main__":

    # trainobject = Indistributiontrain()
    # trainobject.train()
    # trainobject.save()


    # net = classifierood().cuda()
    # indistributionnet = Indistribution().cuda()
    # dataset = mixturedataset(data_test)
    # testlen = 10000
    # traindataset,testdataset = random_split(dataset,(testlen, len(dataset) - testlen))
    # trainloader = DataLoader(traindataset,batch_size=32)
    # for images,reallabels in trainloader:
    #     images = images.cuda()
    #     features,labels,reconstructions = indistributionnet(images)
    #     predresult = net(features,labels,reconstructions)


    detectood = trainforsmoothclassifier()
    detectood.train()
