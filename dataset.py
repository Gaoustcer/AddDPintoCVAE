import torchvision
import torch
from torchvision import datasets,transforms

transform = transforms.ToTensor()
data_train = datasets.MNIST(root = "./data/",
                            transform=transforms.ToTensor(),
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="./data/",
                           transform = transforms.ToTensor(),
                           train = False)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # loader = DataLoader(trainset)
    # for data,labels in loader[:2]:
    #     print(data.shape)
    #     print(labels.shape)
    #     exit()