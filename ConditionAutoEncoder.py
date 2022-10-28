import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from model.model import Encoder,Decoder
import sys
sys.path.append("..")
from dataset import data_test,data_train
class CVAE(object):
    def __init__(self) -> None:
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.trainloader = DataLoader(data_train,batch_size=64)
        self.testloader = DataLoader(data_test,batch_size=1)
        self.optimencoder = torch.optim.Adam(self.encoder.parameters(),lr = 0.001)
        self.optimdecoder = torch.optim.Adam(self.decoder.parameters(),lr = 0.001)
        self.transform = T.ToPILImage()   
        self.lossindex = 0 
        self.writer = SummaryWriter("./logs/loss")

    def translabels(self,labels):
        one_hotlabels = torch.zeros((labels.shape[0],10)).cuda()
        one_hotlabels.scatter_(dim=1,index=labels.unsqueeze(1),src = torch.ones((labels.shape[0],10)).cuda())

        return one_hotlabels
    def train(self):
        for images,labels in tqdm(self.trainloader):
            images = images.cuda()
            labels = labels.cuda()
            labels = self.translabels(labels)
            mu,sigma = self.encoder(images,labels)
            recon_images = self.decoder(sigma,mu,labels)
            different = F.binary_cross_entropy(recon_images.view(-1,28**2),images.view(-1,28**2),reduction='sum')
            # Kldiv = 1 + torch.log(torch.pow(sigma,2)) - torch.pow(sigma,2) - torch.pow(mu,2)
            Kldiv = -0.5 * torch.sum(1 + sigma - torch.pow(mu,2) - torch.exp(sigma))
            # Kldiv = torch.mul(-0.5,Kldiv).mean()
            loss = different + Kldiv
            # loss = different
            self.optimdecoder.zero_grad()
            self.optimencoder.zero_grad()
            loss.backward()
            self.writer.add_scalar('loss',loss,self.lossindex)
            self.lossindex += 1
            self.optimdecoder.step()
            self.optimencoder.step()

    # def valid(self,path):
    #     index = 0
    #     if os.path.exists(path) == False:
    #         os.mkdir(path)
    #     for image,_ in (self.testloader):
    #         if index == 16:
    #             return
    #         image = image.cuda()
    #         mu,sigma = self.encoder(image)
    #         picture = self.decoder(mu,sigma).cpu().detach()
    #         picture = torch.reshape(picture,(28,28))
    #         image = torch.reshape(image,(28,28)).cpu()
    #         picture = self.transform(picture)
    #         image = self.transform(image)
    #         picture.save(os.path.join(path,"VAE{}.png".format(index)))
    #         image.save(os.path.join(path,'origin{}.png'.format(index)))

    #         index+= 1
    
    def deduction(self,path):
        if os.path.exists(path) == False:
            os.mkdir(path)
        for index in range(10):
            mu = torch.zeros((1,16)).to(torch.float32).cuda()
            sigma = torch.ones((1,16)).to(torch.float32).cuda()
            labels = torch.tensor([index]).cuda()
            labels = self.translabels(labels)
            picture = self.decoder(mu,sigma,labels)
            picture = torch.reshape(picture,(28,28)).cpu().detach()
            picture = self.transform(picture)
            picture.save(os.path.join(path,"generate{}.png".format(index)))
        pass

    # def lossfunction(self,sigma,mu)
        