from model.classifier import net
import torch.nn as nn
from torch.utils.data import Dataset
class Combinetwodataset(Dataset):
    def __init__(self,datasetfortrain,datasetforvalidate) -> None:
        super(Combinetwodataset,self).__init__()
        self.images = []
        self.labels = []
        self.uselabels = []
        for image,label in datasetfortrain:
            self.images.append(image)
            self.labels.append(label)
            self.uselabels.append(1)
        for image,label in datasetforvalidate:
            self.images.append(image)
            self.labels.append(label)
            self.uselabels.append(0)
        # print("a image shape is",self.images[0].shape)
        self.images = torch.stack(self.images,dim=0).cuda()
        self.uselabels = torch.tensor(self.uselabels,dtype=torch.int64).cuda()
        # print("images shape",self.images.shape)
        # print("len of labels",len(self.labels))
        # print("len of use labels",len(self.uselabels))
    
    def __len__(self):
        return len(self.uselabels)

    def __getitem__(self, index):
        return self.images[index],self.labels[index],self.uselabels[index]
        # return super().__getitem__(index)

        
class naiveclassifier(nn.Module):
    def __init__(self) -> None:
        super(naiveclassifier,self).__init__()
        self.net = net
        self.identifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5,2)
        )

    def forward(self,images):
        return self.identifier(self.net(images))



from dataset import data_train,data_test
from torch.utils.data import random_split
import torch.nn.functional as F
import torch

class classifierattack(nn.Module):
    def __init__(self,classiferpath = "model/classifier.pkl") -> None:
        super(classifierattack,self).__init__()
        self.targetmodel = torch.load(classiferpath).cuda()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=2,stride=2),
            nn.Flatten()
        )
        # output 36 vector
        self.labelembedding = nn.Sequential(
            nn.Linear(10,16),
            nn.ReLU(),
            nn.Linear(16,32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 + 36,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,2)
        )

    def forward(self,images):
        labels = self.targetmodel(images).detach()
        imageembedding = self.convnet(images)
        labelembedding = self.labelembedding(labels)
        return self.classifier(torch.concat((imageembedding,labelembedding),dim=-1))
from torch.utils.data import DataLoader

# def spilt(dataset,size):
class privacyattackdataset(Dataset):
    def __init__(self,traindataset,testdataset) -> None:
        super(privacyattackdataset,self).__init__()
        self.imagelist = []
        self.trainortest = []
        self.labellist = []
        for image,label in traindataset:
            self.imagelist.append(image)
            self.trainortest.append(1)
            self.labellist.append(label)

        for image,label in testdataset:
            self.imagelist.append(image)
            self.trainortest.append(0)
            self.labellist.append(label)
        
        self.images = torch.stack(self.imagelist,dim = 0)

    def __len__(self):
        return len(self.labellist)

    def __getitem__(self, index):
        return self.images[index],self.labellist[index],self.trainortest[index]
        # return super().__getitem__(index)

class Membershipinfer(object):
    def __init__(self,modelpath:str,traindataset,nottraindataset,validatedataset=None,EPOCH = 32) -> None:
        membershipdataset = privacyattackdataset(traindataset,nottraindataset)
        totallen = len(membershipdataset)
        trainlen = int(0.8 * totallen)
        testlen = totallen - trainlen
        self.trainlen = trainlen
        self.testlen = testlen
        trainmembershipdataset, testmembershipdataset = random_split(membershipdataset,(trainlen,testlen))
        self.membershiploadertrain = DataLoader(trainmembershipdataset,batch_size=32)
        self.membershiploadertest = DataLoader(testmembershipdataset,batch_size=32)
        self.EPOCH = EPOCH
        # self.targetmodel = torch.load(modelpath)
        self.classification = classifierattack(classiferpath=modelpath).cuda()
        # self.classification 
        pass 

    def train(self):
        pass

    def validate(self):
        count = 0
        for images,labels in self.membershiploadertest:
            classificationresult = self.classification(images.cuda().to(torch.float32))
            labels = labels.cuda()
            classificationresult = torch.max(classificationresult,dim=-1).indices
            count += sum(classificationresult == labels)
        return count/self.testlen

class Attacker(object):
    def __init__(self) -> None:
        # 
        self.naiveclassifier = naiveclassifier().cuda()
        # trainlen = len(data_train)
        # testlen = len(data_test)
        totallen = len(data_train) + len(data_test)
        self.batch_size = 32
        self.dataset = Combinetwodataset(data_train,data_test)
        self.traindataset,self.testdataset = random_split(self.dataset,(int(0.8 * totallen),int(0.2*totallen)))
        self.trainloader = DataLoader(self.traindataset,batch_size=self.batch_size,shuffle = True)
        self.testloader = DataLoader(self.testdataset,batch_size=self.batch_size,shuffle=True)
        # self.Truetrain,self.Truetest = random_split(data_train,(int(0.8 * trainlen),trainlen - int(0.8 * trainlen)))
        # self.Falsetrain,self.Falsetest = random_split(data_test,(int(0.8 * testlen),testlen - int(0.8 * testlen)))
        # self.traindataset = Combinetwodataset(self.Truetrain,self.Falsetrain)
        # self.testdataset = Combinetwodataset(self.Truetest,self.Falsetest)
        # self.trainloader = DataLoader(self.traindataset,batch_size=self.batch_size)
        # self.testloader = DataLoader(self.testdataset,batch_size=self.batch_size)
        # self.totaltest = len(self.Truetest)
        # self.totalvalidate = len(self.Falsetest)
        # self.Truetrain = DataLoader(self.Truetrain,batch_size=self.batch_size)
        # self.Truetest = DataLoader(self.Truetest,batch_size=self.batch_size)
        # self.Falsetrain = DataLoader(self.Falsetrain,batch_size=self.batch_size)
        # self.Falsetest = DataLoader(self.Falsetest,batch_size=self.batch_size)
        
        self.classifier = classifierattack().cuda()
        self.epoch = 32
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter("./logs/Attacker")
        self.optim = torch.optim.Adam(self.classifier.parameters(),lr = 0.0001)
    
    from tqdm import tqdm
    
    def validate(self):
        count = 0
        for images,labels,boollabels in self.testloader:
            # images = images.cuda()
            # boollabels = boollabels.cuda().to(torch.int64)
            pred = self.classifier(images)
            indices = torch.max(pred,dim=-1).indices
            count += sum(indices == boollabels)
        return count/len(self.testdataset)

    def train(self):
        from tqdm import tqdm
        index = 0
        for epoch in range(self.epoch):
            for images,labels,boollabels in tqdm(self.trainloader):
                # images = images.cuda()
                # boollabels = boollabels.cuda().to(torch.int64)
                if index % 32 == 0:
                    accrate = self.validate()
                    self.writer.add_scalar("accrate",accrate,index//32)
                self.optim.zero_grad()
                pred = self.classifier(images)
                loss = F.cross_entropy(pred,boollabels)
                loss.backward()
                self.writer.add_scalar("loss",loss,index)
                index += 1
                self.optim.step()


    # def test(self):
    #     count = 0
    #     # totalnumber = le
    #     for truedata,_  in self.Truetest:
    #         truedata = truedata.cuda()
    #         pred = self.classifier(truedata)
    #         index = torch.max(pred,dim=-1).indices
    #         # print("index",index)
    #         # exit()
    #         count += sum(index)
    #     countfalse = 0
    #     for truedata,_  in self.Falsetest:
    #         truedata = truedata.cuda()
    #         pred = self.classifier(truedata)
    #         index = 1 - torch.max(pred,dim=-1).indices
    #         '''
    #         if index[i,0] is largest then it is max
    #         '''
    #         # print("index",index)
    #         # exit()
    #         countfalse += sum(index)
    #     return count/self.totaltest,countfalse/self.totalvalidate
    #     pass

    # def train(self):
    #     from tqdm import tqdm
    #     index = 0
    #     for epo in range(self.epoch):
            
    #         for truelabels,labels in tqdm(self.Truetrain):
    #             if index % 32 == 0:
    #                 acc,accfalse = self.test()
    #                 # print("acc is",acc.shape)
    #                 self.writer.add_scalar("acc",acc,index//32)
    #                 self.writer.add_scalar("falseacc",accfalse,index//32)
    #             '''
    #             for data in train set,we have labels 1 for these images
    #             '''
    #             self.optim.zero_grad()
    #             labels = labels.cuda()
    #             truelabels = truelabels.cuda()
    #             groundtruth = torch.ones(labels.shape,dtype=torch.int64).cuda()
    #             pred = self.classifier(truelabels)
    #             # print("ground truth",groundtruth)
    #             # print("pred",pred)
    #             loss = F.cross_entropy(pred,groundtruth)
    #             loss.backward()
    #             self.optim.step()
    #             index += 1
    #         for falselabels,labels in tqdm(self.Falsetrain):
    #             if index % 32 == 0:
    #                 acc,accfalse = self.test()
    #                 self.writer.add_scalar("acc",acc,index//32)
    #                 self.writer.add_scalar("falseacc",accfalse,index//32)
    #             self.optim.zero_grad()
    #             labels = labels.cuda()
    #             falselabels = falselabels.cuda()
    #             groundtruth = torch.zeros(labels.shape,dtype=torch.int64).cuda()
    #             pred = self.classifier(falselabels)
    #             loss = F.cross_entropy(pred,groundtruth)
    #             loss.backward()
    #             self.optim.step()
    #             index += 1
            
            
        # def forward(self,images):
    #     labels = self.targetmodel(images)
        

    # def train

if __name__ == "__main__":
    for image,label in data_train:
        print(image.shape)
        print(label)
        exit()