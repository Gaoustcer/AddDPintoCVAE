from cvaemodel.oodcvae import cvaeforeachclass
import os
import torch


def trainwithoutnoisewithoutoodmodel():
    log_path = './logs/traincvaeood/nooodnonoise'
    cvaeood = cvaeforeachclass(
        logpath='./logs/traincvaeood/nooodnonoise',
        modelsavepath=os.path.join(log_path,"model"),
        load_oodmodel=False,
        oodfeaturenetpath=None,
        oodclassificationpath=None,
        add_noise=False,
        EPOCH=32,
        use_ood_model=False
    )
    cvaeood.train()
    cvaeood.generateimages()


if __name__ == "__main__":
    trainwithoutnoisewithoutoodmodel()