from ConditionAutoEncoder import CVAE
import os
import torch
import torchvision.transforms as T
from tqdm import tqdm


def _trainCVAE(numberofimages = 1024,rootpath = "generatepicture_DP",logpath = "./logs/traincvae/dp_loss",EPOCH = 8,add_noise = True,\
    use_classification = False):
    transformer = T.ToPILImage()
    ConditionalAutoEncoder = CVAE(add_noise=add_noise,latentspacedim=8,logpath=logpath,use_classifier_for_pretrain=use_classification)
    # EPOCH = 8
    for epoch in range(EPOCH):
        ConditionalAutoEncoder.train()
    # numberofimages = 1024
    # rootpath = "generatepicture_DP"
    if os.path.exists(rootpath) == False:
        os.mkdir(rootpath)
    def imagededuction(label):
        imagepath = os.path.join(rootpath,str(label))
        if os.path.exists(imagepath) == False:
            os.mkdir(imagepath)
        mu = torch.zeros((numberofimages,ConditionalAutoEncoder.latentspacedim)).cuda()
        sigma = torch.ones((numberofimages,ConditionalAutoEncoder.latentspacedim)).cuda()
        labels = torch.ones((numberofimages,),dtype=torch.int64).cuda() * label
        labels = ConditionalAutoEncoder.translabels(labels)
        pictures = ConditionalAutoEncoder.decoder(mu,sigma,labels)[0]
        pictures = torch.reshape(pictures,(-1,28,28))
        # print(pictures.shape)
        for j in tqdm(range(numberofimages)):
            picture = pictures[j]
            
            image = transformer(picture)
            image.save(os.path.join(imagepath,"{}.png".format(j)))
    
    for label in range(10):
        imagededuction(label)


if __name__ == "__main__":
    _trainCVAE(rootpath="generatepicture_withood",logpath="./logs/traincvae/dp_ood",EPOCH=16,add_noise=True,use_classification=True)
    # _trainCVAE(rootpath="generatepicture_DP",logpath="./logs/traincvae/dp_loss",EPOCH=16,add_noise=True)
    # _trainCVAE(rootpath="generatepicture",logpath="./logs/traincvae/loss",EPOCH=8,add_noise=False)
