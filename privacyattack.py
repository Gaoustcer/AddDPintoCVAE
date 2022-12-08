from model.naiveclassifier import Membershipinfer
import torch
from dataset import data_test

if __name__ == "__main__":
    targetmodelpath = "./logs/models/targetmodel"
    traindatapath = "./logs/traindataset"
    testdatapath = "./logs/testdataset"
    traindata = torch.load(traindatapath)
    testdata = torch.load(testdatapath)

    Membershipattack = Membershipinfer(targetmodelpath,traindata,testdata,data_test)
    Membershipattack.train()