import torch
import torch.nn as nn
from torch.utils.data import Dataset

class membershipclassification(nn.Module):
    def __init__(self) -> None:
        super(membershipclassification,self).__init__()
        self.infer = nn.Sequential(
            nn.Linear(10,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,2)
        )
    
    def forward(self,classifcationfeature):
        return self.infer(classifcationfeature)



    