from ConditionAutoEncoder import CVAE
import os
from model.classifier import Trainforclassifier
from model.generateimage import generateimages
from model.classifiergenerate import classifiergenerate
from dataset import data_train,data_test
from torch.utils.data import random_split
import torch
from torchvision import transforms as T
from tqdm import tqdm
# if __name__ == "__main__":
def traincvae():
    Condition_Variation_AutoEncoder = CVAE(add_noise=False,latentspacedim=2)
    EPOCH = 16
    rootpath = "generatepicture"
    transformer = T.ToPILImage()
    # if os.path.exists(rootpath) == False:
    #     os.mkdir(rootpath)
    # Condition_Variation_AutoEncoder.pictureandlabels(128)
    for epoch in range(EPOCH):
        Condition_Variation_AutoEncoder.train()
        # Condition_Variation_AutoEncoder.deduction(rootpath+"/deduction{}".format(epoch))
        Condition_Variation_AutoEncoder.clean()
    numberofimage = 1024 * 8
    
    def _generteimage(numberofimage,label):
        imagepath = os.path.join(rootpath,str(label))
        if os.path.exists(imagepath) == False:
            os.mkdir(imagepath)
        
        mu = torch.zeros((numberofimage,Condition_Variation_AutoEncoder.latentspacedim)).cuda()
        sigma = torch.ones((numberofimage,Condition_Variation_AutoEncoder.latentspacedim)).cuda()
        labels = torch.ones((numberofimage,),dtype=torch.int64).cuda() * label
        labels = Condition_Variation_AutoEncoder.translabels(labels)
        pictures = Condition_Variation_AutoEncoder.decoder(mu,sigma,labels)[0]
        pictures = torch.reshape(pictures,(-1,28,28))
        for j in tqdm(range(numberofimage)):
            picture = pictures[j]
            image = transformer(picture)
            image.save(os.path.join(imagepath,"{}.png".format(j)))
        return pictures
    # _generteimage(10,2)
    # print(pictures.shape)
    for label in range(10):
        _generteimage(numberofimage,label)

    # Condition_Variation_AutoEncoder.generate(numberofimage)

def trainclassifier():
    classifier = Trainforclassifier(testdataset=data_test,traindataset=data_train)
    classifier.train()
    classifier.save()
    pass


def imagegenerate():
    generator = generateimages()
    generator.generatepicture()
def trainclassifiergenerater():
    generateclassifier = classifiergenerate()
    generateclassifier.train()

# from dataset import data_train as traingroundtruth
# from dataset import data_test as validation
# from generatedataset import mydataset

def trainclassificationinrealdataandgeneratedata():
    trainlen = 8000
    testlen = len(mydataset) - trainlen
    trainmydataset,testmydataset = random_split(mydataset,[trainlen,testlen])
    generatebaseline = Trainforclassifier(testmydataset,trainmydataset,logpath="logs/classifier/generatebaseline")
    '''
    test and train in own dataset
    '''
    generatebaseline.train()
    # baseline = Trainforclassifier(validation,traingroundtruth,logpath='logs/classifier/baseline')
    '''
    test and train in real dataset
    '''
    # baseline.train()
    generate = Trainforclassifier(validation,mydataset,logpath='logs/classifier/mydata')
    '''
    train in own dataset
    test in real dataset
    '''
    generate.train()

from model.naiveclassifier import Attacker

def trainattacker():
    attacker = Attacker()
    attacker.train()
if __name__ == "__main__":
    # trainclassifier()
    # trainclassifiergenerater()
    # imagegenerate()
    # traincvae()
    # trainclassificationinrealdataandgeneratedata()
    trainattacker()
    # traincvae()
    # trainclassificationinrealdataandgeneratedata()