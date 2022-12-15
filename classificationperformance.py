# from generatedataset import mydataset
# from generatedataset import dpdataset
from generatedataset import generatedataset
from dataset import data_test,data_train
from model.classifier import Trainforclassifier
from torch.utils.data import random_split
import torch



def _seperate_traindataset():
    len_train = int(0.5 * len(data_train))
    len_test = len(data_train) - len_train
    traindataset,testdataset = random_split(data_train,(len_train,len_test))
    return traindataset,testdataset
def trainanobjectmodel():
    traindataset,testdataset = _seperate_traindataset()
    torch.save(traindataset,"logs/traindataset")
    torch.save(testdataset,"logs/testdataset")
    classifier = Trainforclassifier(testdataset=testdataset,traindataset=traindataset,logpath="logs/classifier/targetmodel",savepath="logs/models/targetmodel",EPOCH=8)
    classifier.train()
    classifier.save()

def _trainclassifierwithgeneratedpdata():
    dpdataset = generatedataset("generatepicture_DP")
    classifier = Trainforclassifier(testdataset=data_test,traindataset=dpdataset,logpath="./logs/classifier/dpclassifier",savepath="logs/models/dpmodel",EPOCH=64)
    classifier.train()
    classifier.save()
    # rootpath = "generatepictur_DP"

    

# import torch
# torch.save()
def trainbaseline():
    # train in real dataset and test in real dataset
    classifier = Trainforclassifier(testdataset=data_test,traindataset=data_train,logpath = "logs/classifier/baseline",savepath="logs/models/realtoreal",EPOCH=8)
    classifier.train()
    classifier.save()

def trainwithgenerate():
    mydataset = generatedataset("generatepicture")
    classifier = Trainforclassifier(testdataset=data_test,traindataset=mydataset,logpath="logs/classifier/generatebaseline",savepath="logs/models/generatetoreal",EPOCH=64)
    classifier.train()
    classifier.save()

def trainwithgenerateood():
    mydataset = generatedataset("generatepicture_withood")
    classifier = Trainforclassifier(testdataset=data_test,traindataset=mydataset,logpath='./logs/classifier/ood',savepath="./logs/models/oodmodel",EPOCH=64)
    classifier.train()
    classifier.save()

if __name__ == "__main__":
    # trainanobjectmodel()

    # trainbaseline()
    # trainwithgenerate()
    # _trainclassifierwithgeneratedpdata()
    trainwithgenerateood()
