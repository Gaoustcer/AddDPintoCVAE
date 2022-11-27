import torchvision
import torch
from torchvision import datasets,transforms


data_train = datasets.MNIST(root = "./data/",
                            transform=transforms.ToTensor(),
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="./data/",
                           transform = transforms.ToTensor(),
                           train = False)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    loader = DataLoader(data_train)
    for data,labels in loader[:2]:
        print(data.shape)
        print(labels.shape)
        exit()