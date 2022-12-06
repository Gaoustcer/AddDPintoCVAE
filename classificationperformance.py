from generatedataset import mydataset
from dataset import data_test,data_train
from model.classifier import Trainforclassifier

def trainbaseline():
    # train in real dataset and test in real dataset
    classifier = Trainforclassifier(testdataset=data_test,traindataset=data_train,logpath = "logs/classifier/baseline",savepath="model/realtoreal",EPOCH=8)
    classifier.train()
    classifier.save()

def trainwithgenerate():
    classifier = Trainforclassifier(testdataset=data_test,traindataset=mydataset,logpath="logs/classsifier/generatebaseline",savepath="model/generatetoreal",EPOCH=8)
    classifier.train()
    classifier.save()

if __name__ == "__main__":
    trainwithgenerate()