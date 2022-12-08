from generatedataset import mydataset
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

    

# import torch
# torch.save()
def trainbaseline():
    # train in real dataset and test in real dataset
    classifier = Trainforclassifier(testdataset=data_test,traindataset=data_train,logpath = "logs/classifier/baseline",savepath="logs/models/realtoreal",EPOCH=8)
    classifier.train()
    classifier.save()

def trainwithgenerate():
    classifier = Trainforclassifier(testdataset=data_test,traindataset=mydataset,logpath="logs/classifier/generatebaseline",savepath="logs/models/generatetoreal",EPOCH=8)
    classifier.train()
    classifier.save()

if __name__ == "__main__":
    # trainanobjectmodel()

    # trainbaseline()
    trainwithgenerate()