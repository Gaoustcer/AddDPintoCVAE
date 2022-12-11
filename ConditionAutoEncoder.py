import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from model.model import Encoder,Decoder
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
from dataset import data_test,data_train
class CVAE(object):
    def __init__(self,add_noise=False,latentspacedim = 8,logpath = "./logs/loss",use_classifier_for_pretrain = False) -> None:
        self.classifier = torch.load("model/classifier.pkl").cuda()
        self.latentspacedim = latentspacedim
        self.encoder = Encoder(add_noise=add_noise,latentspacedim = latentspacedim).cuda()
        self.decoder = Decoder(add_noise=add_noise,latentspacedim = latentspacedim).cuda()
        self.trainloader = DataLoader(data_train,batch_size=256)
        self.testloader = DataLoader(data_test,batch_size=1)
        self.optimencoder = torch.optim.Adam(self.encoder.parameters(),lr = 0.001)
        self.optimdecoder = torch.optim.Adam(self.decoder.parameters(),lr = 0.001)
        self.transform = T.ToPILImage()
        self.alpha = 1
        self.beta = 1
        '''
        loss = MSE_loss(images) + alpha * KLdiv + beta * classifierloss
        '''   
        self.lossindex = 0 
        self.gradlogindex = 0
        self.writer = SummaryWriter(logpath)
        # self.encodergradientlog = []
        self.generatepicturenumber = 128 ** 2
        self.colors = plt.get_cmap('RdBu',10)
        # for index,param in enumerate(self.encoder.parameters()):
        #     self.encodergradientlog.append(SummaryWriter("./logs/gradient/encoder/grad{}".format(index)))
        # for index,param in enumerate(self.decoder.parameters()):
        #     self.encodergradientlog.append(SummaryWriter("./logs/gradient/decoder/grad{}".format(index)))
        self.trainindex = 0

    def translabels(self,labels):
        one_hotlabels = torch.zeros((labels.shape[0],10)).cuda()
        one_hotlabels.scatter_(dim=1,index=labels.unsqueeze(1),src = torch.ones((labels.shape[0],10)).cuda())

        return one_hotlabels
    
    def scatter(self,latent_space,labels):
        plt.scatter(x = latent_space.detach().cpu()[:,0],y = latent_space.detach().cpu()[:,1],c = self.colors(labels.detach().cpu()))
    
    def clean(self):
        plt.savefig("distributions/pic{}.png".format(self.trainindex))
        self.trainindex += 1
        plt.figure()

    def save(self):
        torch.save(self.encoder,'model/encoder')
        torch.save(self.decoder,'model/decoder')
    
    def pictureandlabels(self,size:int,path = "Picture/picturegenerate"):
        labels = torch.randint(0,10,(size,)).cuda()
        onehotlabels = self.translabels(labels)
        # for index in 
        mu = torch.zeros((size,self.latentspacedim)).to(torch.float32).cuda()
        sigma = torch.ones((size,self.latentspacedim)).to(torch.float32).cuda()
        # print("onehot labels",onehotlabels.shape)
        picture = self.decoder(sigma,mu,onehotlabels)[0]
        picture = torch.reshape(picture,(size,28,28)).cpu().detach()
        for index in range(size):
            image = self.transform(picture[index])
            image.save(path+"/{}.png".format(index))
        
        import numpy as np
        labels = labels.detach().cpu().numpy()
        np.save(path + "/labels.npy",labels)

        # onhotlabels = torch.zeros((size,10)).cuda()
        # onhotlabels.scatter_(dim=1,index=labels.unsqueeze(1),src = torch.ones((size,10)).cuda())
    def generate(self,numberperlabel):
        rootpath = 'generatepicture'
        for i in range(10):
            rootpath = "generatepicture/{}".format(i)
            if os.path.exists(rootpath) == False:
                os.mkdir(rootpath)
            labels = self.translabels(torch.ones((1,),dtype=torch.int64).cuda() * numberperlabel)

            # for j in range(numberperlabel):
            # for j in range(numberperlabel):
            
            mu = torch.zeros((numberperlabel,self.latentspacedim)).cuda()
            sigma = torch.ones((numberperlabel,self.latentspacedim)).cuda()
                # print(mu.shape)
                # print(sigma.shape)
                # labels = self.translabels(torch.ones((1,),dtype=torch.int64).cuda() * numberperlabel)
            pictures = self.decoder(sigma,mu,labels)[0]
                # print("picture shape is",pictures.shape)
                # for j in range(numberperlabel):
            for j in range(numberperlabel):
                tensorpicture = torch.reshape(pictures[0],(28,28)).detach().cpu()
                image = self.transform(tensorpicture)
                image.save(os.path.join(rootpath,"{}.png".format(j)))

        # labels = torch.randint(0,10,(self.generatepicturenumber,)).cuda()
        # mu = torch.zeros((self.generatepicturenumber,self.latentspacedim)).cuda()
        # sigma = torch.ones((self.generatepicturenumber,self.latentspacedim)).cuda()
        # labels = self.translabels(labels)
        # pictures = self.decoder.forward(sigma,mu,labels)[0]
        # pictures = torch.reshape(pictures,(self.generatepicturenumber,28,28)).detach().cpu()
        # for i in range(self.generatepicturenumber):
        #     picture = pictures[i]
        #     image = self.transform(picture)
        #     image.save("generatepicture/{}.png".format(i))
        # labels = labels.cpu().detach().numpy()
        # import numpy as np

        # np.save("./generatepicture/label.npy",labels)
    def train(self):
        for images,labels in tqdm(self.trainloader):
            images = images.cuda()
            originlabels = labels
            labels = labels.cuda()
            labels = self.translabels(labels)
            mu,sigma = self.encoder(images,labels)
            recon_images,feature = self.decoder(sigma,mu,labels)
            self.scatter(feature,originlabels)
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
            # self.lossindex()
            self.loggradient()
            # self.scatter()

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
    def loggradient(self):
        for index,param in enumerate(self.encoder.parameters()):
            self.writer.add_histogram("encoder_{}_grad".format(index),param.grad.cpu().data.numpy(),self.gradlogindex)
            self.writer.add_histogram("encoder_{}_data".format(index),param.cpu().data.numpy(),self.gradlogindex)
        for index,param in enumerate(self.decoder.parameters()):
            self.writer.add_histogram("decoder_{}_grad".format(index),param.grad.cpu().data.numpy(),self.gradlogindex)
            self.writer.add_histogram("decoder_{}_data".format(index),param.cpu().data.numpy(),self.gradlogindex)
            
        self.gradlogindex += 1
        pass 
    def deduction(self,path):
        if os.path.exists(path) == False:
            os.mkdir(path)
        for index in range(10):
            mu = torch.zeros((1,self.latentspacedim)).to(torch.float32).cuda()
            sigma = torch.ones((1,self.latentspacedim)).to(torch.float32).cuda()
            labels = torch.tensor([index]).cuda()
            labels = self.translabels(labels)
            picture = self.decoder(mu,sigma,labels)[0]
            picture = torch.reshape(picture,(28,28)).cpu().detach()
            picture = self.transform(picture)
            picture.save(os.path.join(path,"generate{}.png".format(index)))
        pass

    # def lossfunction(self,sigma,mu)
        