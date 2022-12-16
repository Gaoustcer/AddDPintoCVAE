# from model.naiveclassifier import Membershipinfer
import torch
import torch.nn.functional as F
import torch.nn as nn
# from dataset import data_test
from generatedataset import generatedataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
N = 280

def testdatsettrainiid(rootpath = "generatepicture_withood"):
    dataset = generatedataset(rootpath)
    loader = DataLoader(dataset)


if __name__ == "__main__":
    net1 = nn.Sequential(
        nn.Linear(4,2),
        nn.ReLU(),
        nn.Linear(2,1)
    )
    net2 = nn.Sequential(
        nn.Linear(1,4),
        nn.ReLU(),
        nn.Linear(4,1)
    )
    tensor = torch.rand(32,4)
    result = net1(tensor)
    print(result)
    with torch.no_grad():
        output = net2(result)
        output = torch.sum(output)
    print(output)
    output.backward()
    # output = torch.sum(output)

    # features = torch.rand(23,1,40)
    # transcov = nn.ConvTranspose1d(in_channels=1,out_channels=1,kernel_size=4,stride=2,padding=1)
    # result = transcov(features)


    # result = transcov(result)
    # result = torch.reshape(result,(23,1,16,10))
    # twotransconv = nn.ConvTranspose2d(
    #     in_channels=1,out_channels=1,
    #     kernel_size=(3,4),
    #     stride=(2,3),
    #     padding=(3,2),
    #     output_padding=1)

    # print(result.shape) # (16,10)\to (28,28)
    # print(twotransconv(result).shape)
    # images = torch.rand(N,1,28,28)
    # labels = torch.rand()
    # dataset = TensorDataset(images,labels)
    # loader = DataLoader(dataset,batch_size=32)
    # for image,label in loader:
    #     print(image.shape)
    #     print(label.shape)
    #     exit()
    # tensor = torch.rand(3,2)
    # softmaxlayer = nn.Softmax()
    # softmaxtensor = softmaxlayer(tensor)
    # label = torch.randint(0,2,3)
    # print(label)
    # crossentropyloss = F.cross_entropy(softmaxtensor,label)
    # targetmodelpath = "./logs/models/targetmodel"
    # traindatapath = "./logs/traindataset"
    # testdatapath = "./logs/testdataset"
    # traindata = torch.load(traindatapath)
    # testdata = torch.load(testdatapath)

    # Membershipattack = Membershipinfer(targetmodelpath,traindata,testdata,data_test)
    # Membershipattack.train()