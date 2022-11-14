import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
class generatenet(nn.Module):
    def __init__(self,noise_dim = 32,onehot_dim = 10,embed_dim = 32,picture_dim = 28 * 28) -> None:
        super(generatenet,self).__init__()
        self.noiseencoder = nn.Sequential(
            nn.Linear(noise_dim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,embed_dim)
        )
        self.onhotencoder = nn.Sequential(
            nn.Linear(onehot_dim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,embed_dim)            
        )

        self.encoderaspicture = nn.Sequential(
            nn.Linear(2 * embed_dim,32),
            nn.ReLU(),
            nn.Linear(32,picture_dim),
            nn.Sigmoid()
        )
        self.noisedim = noise_dim
        self.numberlabels = onehot_dim
        self.picwidth = 28
    def forward(self,labels):
        
        randomnoise = torch.randn((len(labels),self.noisedim)).cuda()
        onehotembedding = F.one_hot(labels,self.numberlabels).cuda().to(torch.float32)
        noiseembedding = self.noiseencoder(randomnoise)
        labelembedding = self.onhotencoder(onehotembedding)
        embedding = torch.concat((noiseembedding,labelembedding),dim=-1).cuda()
        picture = self.encoderaspicture(embedding)
        return torch.reshape(picture,(len(labels),28,28))
        pass
class classifiergenerate(object):
    def __init__(self,path = "model/classifier.pkl") -> None:
        self.classifier = torch.load(path).cuda()
        self.decoder = generatenet().cuda()
        self.optim = torch.optim.Adam(self.decoder.parameters(),lr = 0.0001)
        self.batchsize = 32
        self.epoch = 128
        self.trainperepoch = 128
        self.writer = SummaryWriter("./logs/classifiergenerate")
        self.lossindex = 0

    def train(self):
        from tqdm import tqdm
        for epoch in range(self.epoch):
            for _ in tqdm(range(self.trainperepoch)):
                labels = torch.randint(0,10,(self.batchsize,)).cuda()
                picture = self.decoder(labels)
                prob = self.classifier(picture.unsqueeze(1))
                self.optim.zero_grad()
                loss = F.cross_entropy(prob,labels)
                loss.backward()
                self.writer.add_scalar("loss",loss,self.lossindex)
                self.lossindex += 1
                self.optim.step()
            self.save(epoch)
            
    
    def save(self,epo):
        torch.save(self.decoder,'model/generate/generate{}.pkl'.format(epo))
        
if __name__ == "__main__":
    generate = generatenet().cuda()
    labels = torch.randint(0,10,(17,)).cuda()
    picture = generate(labels)
    classifier = torch.load("classifier.pkl").cuda()
    print(picture.shape)
    picture = picture.unsqueeze(1)
    predlabel = classifier(picture)
    print(predlabel.shape)
