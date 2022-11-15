from ConditionAutoEncoder import CVAE
import os
from model.classifier import Trainforclassifier
from model.generateimage import generateimages
from model.classifiergenerate import classifiergenerate
from dataset import data_train,data_test
# if __name__ == "__main__":
def traincvae():
    Condition_Variation_AutoEncoder = CVAE(add_noise=False,latentspacedim=2)
    EPOCH = 32
    rootpath = "picture"
    if os.path.exists(rootpath) == False:
        os.mkdir(rootpath)
    for epoch in range(EPOCH):
        Condition_Variation_AutoEncoder.train()
        Condition_Variation_AutoEncoder.deduction(rootpath+"/deduction{}".format(epoch))
        Condition_Variation_AutoEncoder.clean()
        

def trainclassifier():
    classifier = Trainforclassifier(testdataset=data_test,traindataset=data_train)
    classifier.train()
    classifier.save()
    pass


def imagegenerate():
    generator = generateimages()
    generator.generatepicture()
def trainclassifiergenerater():
    generateclassifier = classifiergenerate()
    generateclassifier.train()
if __name__ == "__main__":
    # trainclassifier()
    # trainclassifiergenerater()
    imagegenerate()