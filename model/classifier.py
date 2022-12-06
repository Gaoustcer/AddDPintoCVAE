import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter


net = nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=1,out_channels=1,kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(36,32),
    nn.ReLU(),
    nn.Linear(32,10)
    # nn.Softmax()
)
class Trainforclassifier(object):
    def __init__(self,testdataset,traindataset,logpath,savepath = "model/classifier.pkl",EPOCH = 16) -> None:
        self.net = net.cuda()
        self.savepath = savepath
        self.testloader = DataLoader(testdataset,batch_size=32,shuffle=True)
        self.trainloader = DataLoader(traindataset,batch_size=64,shuffle=True)
        self.testlen = len(testdataset)
        self.trainlen = len(traindataset)
        self.EPOCH = EPOCH
        self.optim = torch.optim.Adam(self.net.parameters(),lr = 0.0001)
        self.index = 0
        self.writer = SummaryWriter(logpath)

    def train(self):
        for epoch in range(self.EPOCH):
            from tqdm import tqdm
            for images,labels in tqdm(self.trainloader):
                if self.index % 32 == 0:
                    accrate = self.validate()
                    self.writer.add_scalar("accurate",accrate,self.index)
                images = images.cuda()
                labels = labels.cuda()
                # print('shape is',images.shape)
                # if images.shape[1] != 1:
                #     images = images.unsqueeze(1)
                # print("images shape is",images.shape)
                # print("labels shape is",labels.shape)
                predlabels = self.net(images)
                # print(pred)
                # prob = torch.gather(predlabels,dim=1,index=labels.unsqueeze(-1))
                self.optim.zero_grad()
                # print("images shape",images.shape)
                # print("predlabels",predlabels[0])
                # print("lebels is ",labels)
                # return
                # exit()
                loss = torch.nn.functional.cross_entropy(predlabels,labels)
                # loss = torch.sum(torch.log(prob))
                loss.backward()
                self.writer.add_scalar("loss",loss,self.index)
                self.optim.step()
                # accrate = self.validate()
                
                self.index += 1
            # print(f"acc rate for epoch{epoch},{accrate}")                
            
        pass

    def validate(self):
        accnum = 0
        for images,labels in self.testloader:
            images = images.cuda()
            labels = labels.cuda()
            predlabels = torch.argmax(self.net(images),dim=1).squeeze()
            accnum += torch.sum(predlabels == labels)
        return accnum/self.testlen
    

    def save(self):
        torch.save(self.net,self.savepath)
