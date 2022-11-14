import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader

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
    def __init__(self,testdataset,traindataset) -> None:
        self.net = net.cuda()
        self.testloader = DataLoader(testdataset,batch_size=32)
        self.trainloader = DataLoader(traindataset,batch_size=64)
        self.testlen = len(testdataset)
        self.trainlen = len(traindataset)
        self.EPOCH = 32
        self.optim = torch.optim.Adam(self.net.parameters(),lr = 0.0001)

    def train(self):
        for epoch in range(self.EPOCH):
            from tqdm import tqdm
            for images,labels in tqdm(self.trainloader):
                images = images.cuda()
                labels = labels.cuda()
                predlabels = self.net(images)
                # print(pred)
                # prob = torch.gather(predlabels,dim=1,index=labels.unsqueeze(-1))
                self.optim.zero_grad()
                loss = torch.nn.functional.cross_entropy(predlabels,labels)
                # loss = torch.sum(torch.log(prob))
                loss.backward()
                self.optim.step()
            accrate = self.validate()
            print(f"acc rate for epoch{epoch},{accrate}")                
            
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
        torch.save(self.net,"model/classifier.pkl")
