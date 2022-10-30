from ConditionAutoEncoder import CVAE
import os

if __name__ == "__main__":
    Condition_Variation_AutoEncoder = CVAE(add_noise=True)
    EPOCH = 32
    rootpath = "picturegaussian"
    if os.path.exists(rootpath) == False:
        os.mkdir(rootpath)
    for epoch in range(EPOCH):
        Condition_Variation_AutoEncoder.train()
        Condition_Variation_AutoEncoder.deduction(rootpath+"/deduction{}".format(epoch))
        