from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms as T
import torch
import numpy as np


class generatedataset(Dataset):
    def __init__(self) -> None:
        super(generatedataset,self).__init__()
        self.trans = T.ToTensor()
        self.rootpath = "generatepicture/"
        Imagelist = []
        self.labellist = []
        from tqdm import tqdm
        for subpathname in os.listdir(self.rootpath):
            labelpath = os.path.join(self.rootpath,subpathname)
            for filename in tqdm(os.listdir(labelpath)):
                filepath = os.path.join(labelpath,filename)
                image = Image.open(filepath)
                tensor = self.trans(image)
                Imagelist.append(tensor)
            self.labellist += [int(subpathname)] * len(os.listdir(labelpath))
        self.images = torch.stack(Imagelist,dim = 0)
        
            

                
        # imagetensorlist = []
        # self.labels = np.load(os.path.join(self.rootpath,"labels.npy"))
        # for index in range(len(self.labels)):
        #     filename = "{}.png".format(index)
        #     filepath = os.path.join(self.rootpath,filename)
        #     image = Image.open(filepath)
        #     tensor = self.trans(image)
        #     imagetensorlist.append(tensor)
        # for filename in os.listdir(self.rootpath):
        #     if "png" in filename:
        #         filepath = os.path.join(self.rootpath,filename)
        #         image = Image.open(filepath)
        #         tensor = self.trans(image)
        #         imagetensorlist.append(tensor)
        #     if "labels.npy" in filename:
        #         import numpy as np  
        #         self.labels = np.load(os.path.join(self.rootpath,filename))
        # self.images = torch.concat(imagetensorlist,dim=0)
        # self.images = torch.stack(imagetensorlist,dim=0)
    
    def __len__(self):
        return len(self.labellist)
    
    def __getitem__(self, index):
        return self.images[index],self.labellist[index]
mydataset = generatedataset()

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    loader = DataLoader(generatedataset(),batch_size=32,shuffle=True)
    for image,label in loader:
        print(image.shape)
        print(label.shape)
        print(label)
        exit()
        # return super().__getitem__(index)
        
