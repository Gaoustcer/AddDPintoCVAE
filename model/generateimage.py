import torchvision.transforms as T
import torch
class generateimages(object):
    def __init__(self,path = "model/generate/generate739.pkl",savepath = 'pictureclassifier/') -> None:
        self.net = torch.load(path).cuda()
        self.transformer = T.ToPILImage()
        self.savepath = savepath

    def generatepicture(self):
        labels = torch.tensor(range(10)).cuda()
        pictures = self.net(labels)
        for i in range(10):
            picture = pictures[i]
            image = self.transformer(picture)
            image.save(self.savepath + f"{i}.png")
