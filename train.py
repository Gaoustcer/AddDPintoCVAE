from ConditionAutoEncoder import CVAE


if __name__ == "__main__":
    Condition_Variation_AutoEncoder = CVAE()
    EPOCH = 32
    rootpath = "picture"
    for epoch in range(EPOCH):
        Condition_Variation_AutoEncoder.train()
        Condition_Variation_AutoEncoder.deduction(rootpath+"/deduction{}".format(epoch))
        