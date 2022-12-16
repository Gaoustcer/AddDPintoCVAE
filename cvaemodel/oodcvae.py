import torch.nn as nn

import torch
from torch.utils.data import DataLoader

from model.model import Encoder,Decoder
from torch.utils.tensorboard import SummaryWriter
from model.smood import iddata as imagefeaturenet
from model.smood import classifierood as oodclassification
# from dataset import data_train
from dataset import data_train
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision import transforms as T


class cvaeforeachclass(object):
    def __init__(self,
        logpath, # path to write logs
        modelsavepath, # path to save model
        load_oodmodel,
        oodfeaturenetpath,
        oodclassificationpath,
        latent_dim = 8,
        add_noise = True,
        data_train = data_train,
        EPOCH = 32,
        use_ood_model = False
    ) -> None:
        self.transformer = T.ToPILImage()
        self.useoodmodel = use_ood_model
        self.latentdim = latent_dim
        self.encoder = Encoder(add_noise=False,latentspacedim=latent_dim).cuda()
        self.decoder = Decoder(add_noise=add_noise,latentspacedim=latent_dim).cuda()
        self.traindata = data_train
        self.trainloader = DataLoader(self.traindata,batch_size=32)
        if load_oodmodel:
            self.featurenet = torch.load(oodfeaturenetpath).cuda()
            self.oodclassificationnet = torch.load(oodclassificationpath).cuda()
        else:
            self.featurenet = imagefeaturenet().cuda()
            self.oodclassificationnet = oodclassification().cuda()
        self.logpath = logpath
        self.modelsavedpath = modelsavepath
        self.writer = SummaryWriter(self.logpath)
        self.logindex = 0
        self.EPOCH = EPOCH
        self.optimencoder = torch.optim.Adam(self.encoder.parameters(),lr = 0.0001)
        self.optimdecoder = torch.optim.Adam(self.decoder.parameters(),lr = 0.0001) 
        self.optimfeature = torch.optim.Adam(self.featurenet.parameters(),lr = 0.0001)
        self.optimoodclassification = torch.optim.Adam(self.oodclassificationnet.parameters(),lr = 0.0001)
    
    def zero_grad(self):
        self.optimdecoder.zero_grad()
        self.optimencoder.zero_grad()
        self.optimfeature.zero_grad()
        self.optimoodclassification.zero_grad()
    
    def step(self):
        self.optimdecoder.step()
        self.optimencoder.step()
        self.optimfeature.step()
        self.optimoodclassification.step()

    def _KLdivloss(self,mu:torch.tensor,sigma:torch.Tensor):
        return -0.5 * torch.sum(1 + sigma - torch.pow(mu,2) - torch.exp(sigma))

    def trainforanepoch(self):
        from tqdm import tqdm
        for images,labels in tqdm(self.trainloader):
            self.zero_grad()
            images = images.cuda()
            labels = labels.cuda()
            mu,sigma = self.encoder.forward(images,labels)
            reconstructimages = self.decoder.forward(sigma,mu,labels)[0]
            '''
            use in distribution data train the feature extraction
            '''
            realfeature,realclassification,realconstruction = self.featurenet(images)
            classificationloss = F.cross_entropy(realclassification,labels)
            reconstructionloss = F.binary_cross_entropy(realconstruction,images)
            featureloss = classificationloss + reconstructionloss
            featureloss.backward()
            self.optimfeature.step()

            '''
            combine indistribution(1) and ood(0) to train classifier
            '''
            if self.useoodmodel:
                indistribution = torch.ones(len(labels)).cuda()
                ooddistribution = torch.zeros(len(labels)).cuda()
                oodclassifierdataset = TensorDataset(
                    torch.concat((images,reconstructimages),dim=0),
                    torch.concat((indistribution,ooddistribution),dim=0))
                oodloader = DataLoader(oodclassifierdataset,batch_size=16,shuffle=True)
                for oodimages,whetheroodlabels in oodloader:
                    self.optimoodclassification.zero_grad()
                    with torch.no_grad():
                        features,labelpreds,decodeimages = self.featurenet(oodimages)
                    predoodlabels = self.oodclassificationnet(features,labelpreds,decodeimages)
                    predoodloss = F.cross_entropy(predoodlabels,whetheroodlabels)
                    predoodloss.backward()
                    self.optimoodclassification.step()

            '''
            Kl divergence loss and reconstruction loss
            '''            
            klloss = self._KLdivloss(sigma=sigma,mu=mu)
            # encoderpicture = self.decoder.forward()
            reconstructloss = F.binary_cross_entropy(reconstructimages,images)
            # self.classifier 
            '''
            classification loss
            '''
            generateimagefeatures, generateimageclassification, generateimages = self.featurenet(reconstructimages)
            labelloss = F.cross_entropy(generateimageclassification,labels)
            oodprediction = self.oodclassificationnet(generateimagefeatures,generateimageclassification,generateimages)
            if self.useoodmodel:
                oodloss = F.cross_entropy(oodprediction,torch.ones_like(labels))
                loss = klloss + labelloss + oodloss + reconstructloss
                self.writer.add_scalars("loss",{'klloss':klloss,'labelsloss':labelloss,"oodloss":oodloss,"reconstloss":reconstructloss},global_step=self.logindex)
            else:
                loss = klloss + reconstructloss
                self.writer.add_scalars("loss",{"klloss":klloss,'labelsloss':labelloss,'reconstructloss':reconstructloss},global_step=self.logindex)
            self.logindex += 1
            # reconstructloss = loss + reconstructloss
            loss.backward()
            self.optimencoder.step()
            self.optimdecoder.step()

            
            '''
            reconstruct images are treated as ood data
            images are treated as in-distribution data
            '''
            
            
        # pass
    
    def save(self):
        import os
        if os.path.exists(self.modelsavedpath) == False:
            os.mkdir(self.modelsavedpath)
        torch.save(self.encoder,os.path.join(self.modelsavedpath,"encoder"))
        torch.save(self.decoder,os.path.join(self.modelsavedpath,"decoder"))
        torch.save(self.featurenet,os.path.join(self.modelsavedpath,"feature"))
        torch.save(self.oodclassificationnet,os.path.join(self.modelsavedpath,"oodclassification"))
        
    def generateimages(self,numberperimage = 1024):
        import os
        rootpath = os.path.join(self.logpath,"picture")
        if os.path.exists(rootpath) == False:
            os.mkdir(rootpath)
        for id in range(10):
            localpath = os.path.join(rootpath,str(id))
            if os.path.exists(localpath) == False:
                os.mkdir(localpath)
            labels = torch.ones((numberperimage,),dtype=torch.int64,device="cuda") * id
            images = self._generateimages(labels).cpu()
            # print(images.shape)
            for i in range(numberperimage):
                path = os.path.join(localpath,str(i)+".png")
                image = images[i]
                # print(image.shape)
                image = self.transformer(image)
                image.save(path)

            
    
    def _generateimages(self,labels):
        # numberofimages = len(labels)
        with torch.no_grad():
            images = self.decoder.forward(mu=None,sigma=None,labels = labels)[0]
        return images
        # mu = torch.ones((numberofimages,self.latentdim),dtype=torch.)

    
    def train(self):
        for epoch in range(self.EPOCH):
            self.trainforanepoch()
        self.save()

    # pass
