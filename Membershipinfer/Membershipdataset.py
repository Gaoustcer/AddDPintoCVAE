from torch.utils.data import Dataset
import torch



class Membershipdataset(Dataset):
    def __init__(self,traindatasetpath = "./logs/traindataset",testdatasetpath = "./logs/testdataset") -> None:
        super(Membershipdataset,self).__init__()
        traindataset = torch.load(traindatasetpath)
        testdataset = torch.load(testdatasetpath)
        images = []
        labels = []
        for image,_ in traindataset:
            images.append(image.cuda())
            labels.append(1)
        for image,_ in testdataset:
            images.append(image.cuda())
            labels.append(0)
        self.image = torch.stack(images,dim=0)
        self.labels = torch.tensor(labels,dtype=torch.int64).cuda()
        print("label sum is",sum(self.labels))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.image[index],self.labels[index]
        # return super().__getitem__(index)
