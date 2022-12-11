from ConditionAutoEncoder import CVAE
import os
import torch
import torchvision.transforms as T
from tqdm import tqdm


def _trainCVAE():
    transformer = T.ToPILImage()
    ConditionalAutoEncoder = CVAE(add_noise=True,latentspacedim=8)
    EPOCH = 16
    for epoch in range(EPOCH):
        ConditionalAutoEncoder.train()
    numberofimages = 1024 * 8
    rootpath = "generatepicture_DP"
    if os.path.exists(rootpath) == False:
        os.mkdir(rootpath)
    def imagededuction(label):
        imagepath = os.path.join(rootpath,str(label))
        if os.path.exists(imagepath):
            os.mkdir(imagepath)
        mu = torch.zeros((numberofimages,ConditionalAutoEncoder.latentspacedim)).cuda()
        sigma = torch.ones((numberofimages,ConditionalAutoEncoder.latentspacedim)).cuda()
        labels = torch.ones((numberofimages,),dtype=torch.int64).cuda() * label
        labels = ConditionalAutoEncoder.translabels(labels)
        pictures = ConditionalAutoEncoder.decoder(mu,sigma,labels)[0]
        pictures = torch.reshape(pictures,(-1,28,28))
        for j in tqdm(range(numberofimages)):
            picture = pictures[j]
            image = transformer(pictures)
            image.save(os.path.join(imagepath,"{}.png".format(j)))
    
    for label in range(10):
        imagededuction(label)


if __name__ == "__main__":
    _trainCVAE()