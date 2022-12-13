import torch

from torch.utils.data import DataLoader
from generatedataset import generatedataset

dataset = generatedataset("generatepicture_DP")
net = torch.load("model/classifier.pkl").cuda()
loader = DataLoader(dataset,batch_size=64,shuffle=True)
for data,label in loader:
    data = data.cuda()
    predlabel = net(data)
    predlabel = torch.max(predlabel,dim=-1).indices
    label = label.cuda()
    print(sum(label == predlabel))
    print(label)
    exit()