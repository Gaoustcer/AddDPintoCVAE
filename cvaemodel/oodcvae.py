import torch.nn as nn

import torch
from torch.utils.data import DataLoader

from model.model import Encoder,Decoder
from torch.utils.tensorboard import SummaryWriter
from model.smood import iddata as imagefeaturenet
from model.smood import classifierood as oodclassification
# from dataset import data_train
from dataset import data_train




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
        EPOCH = 32
    ) -> None:
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
        self.EPOCH = EPOCH
        self.optimencoder = torch.optim.Adam(self.encoder.parameters(),lr = 0.0001)
        self.optimdecoder = torch.optim.Adam(self.decoder.parameters(),lr = 0.0001) 
        self.optimfeature = torch.optim.Adam(self.featurenet.parameters(),lr = 0.0001)
        self.optimoodclassification = torch.optim.Adam(self.oodclassificationnet.parameters(),lr = 0.0001)
    
    def trainforanepoch(self):
        from tqdm import tqdm
        for images,labels in tqdm(self.trainloader):
            images = images.cuda()
            labels = images.cuda()
            mu,sigma = self.encoder.forward(images,labels)

            
        # pass
    
    def save(self):
        import os
        torch.save(self.encoder,os.path.join(self.logpath,"encoder"))
        torch.save(self.decoder,os.path.join(self.logpath,"decoder"))
        torch.save(self.featurenet,os.path.join(self.logpath,"feature"))
        torch.save(self.oodclassificationnet,os.path.join(self.logpath,"oodclassification"))
        

    
    def train(self):
        for epoch in range(self.EPOCH):
            self.trainforanepoch()
        self.save()

    # pass
