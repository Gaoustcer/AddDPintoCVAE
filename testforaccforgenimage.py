import torch
from generatedataset import mydataset
from dataset import data_test
from torch.utils.data import DataLoader
classifier = torch.load("model/classifier.pkl").cuda()
def validate(dataset):
    totaldataset = DataLoader(dataset,batch_size=32)
    totallen = len(dataset)
    samenumber = 0
    for images,labels in totaldataset:
        predlabels = classifier(images)
        
