# from model.naiveclassifier import Membershipinfer
import torch
import torch.nn.functional as F
import torch.nn as nn
# from dataset import data_test

if __name__ == "__main__":
    tensor = torch.rand(3,2)
    softmaxlayer = nn.Softmax()
    softmaxtensor = softmaxlayer(tensor)
    label = torch.randint(0,2,3)
    print(label)
    # targetmodelpath = "./logs/models/targetmodel"
    # traindatapath = "./logs/traindataset"
    # testdatapath = "./logs/testdataset"
    # traindata = torch.load(traindatapath)
    # testdata = torch.load(testdatapath)

    # Membershipattack = Membershipinfer(targetmodelpath,traindata,testdata,data_test)
    # Membershipattack.train()