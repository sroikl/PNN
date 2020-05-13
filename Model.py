import torch
from torch import nn
from torchvision.models import inception_v3
from TCN.tcn import TemporalConvNet

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        self.inception = inception_v3(pretrained=True, transform_input=True, aux_logits=True)
        self.inception.fc = Identity()
        self.inception.eval()

        for p in self.inception.parameters():
            p.requires_grad = False

    # make sure that the inception model stays on eval
    def train(self, mode=True):
        return self

    def forward(self, x: torch.Tensor):
        """

        :param x: a batch of image sequences of either size (NxTxCxHxW) or the squeezed size (NxTxHxW)
        :return: a batch of feature vectors for each image of size (NxTx2048)
        """

        # if we got a batch of sequences we have to calculate each sequence separately
        N, T = x.shape[:2]
        # return self.inception(x.view(-1, *x.shape[2:])).view(N, T, -1)
        return self.inception(x.reshape(N*T,*x.shape[2:])).view(N, T, -1)

    def _prepare_for_gpu(self):
        if torch.cuda.is_available():
            self.inception.cuda()

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class TCN(TemporalConvNet):
    def __init__(self, num_levels: int = 3, num_hidden: int = 600, embedding_size: int = 128, kernel_size=2,
                 dropout=0.2):
        """

        :param num_levels: the number of TCN layers
        :param num_hidden: number of Feature Maps used in the hidden layers
        :param embedding_size: size of final feature vector
        :param kernel_size: kernel size, make sure that it matches the feature vector size
        :param dropout: dropout probability
        :return: a TemporalConvNet matching the inputted params
        """
        num_channels = [num_hidden] * (num_levels - 1) + [embedding_size]
        super().__init__(2048, num_channels, kernel_size, dropout)
        self.net= TemporalConvNet(num_inputs= 2048, num_channels= num_channels, kernel_size= kernel_size, dropout= dropout)

    def forward(self, x: torch.Tensor):
        """

        :param x: input tensor of size (NxTx2048),
                where N is the batch size, T is the sequence length and 2048 is the input embedding dim
        :return: tensor of size (N x T x embedding_size) where out[:,t,:] is the output given all values up to time t
        """

        # transpose each sequence so that we get the correct size for the TemporalConvNet
        x = torch.stack([m.t() for m in x])
        out = self.net(x)

        # undo the previous transpose
        return torch.stack([m.t() for m in out])

class Classifier(nn.Module):
    def __init__(self,embedding_size):
        super(Classifier,self).__init__()
        self.classifier= nn.Sequential(
                        nn.Linear(in_features=embedding_size,out_features=512),nn.LeakyReLU(negative_slope=0.1),
                        nn.Linear(in_features=512,out_features= 128),nn.LeakyReLU(negative_slope=0.1),
                        nn.Linear(in_features=128,out_features=32),nn.LeakyReLU(negative_slope=0.1),
                        nn.Linear(in_features= 32, out_features= 1)
        )
    def forward(self, x):
        return self.classifier(x)

class TCN_Model(nn.Module):
    def __init__(self,num_levels: int, num_hidden: int , embedding_size: int, kernel_size, dropout):
        super().__init__()

        self.TCN= TCN(num_levels=num_levels,num_hidden=num_hidden,embedding_size=embedding_size,kernel_size=kernel_size,
                      dropout=dropout)

        self.Classifier= Classifier(embedding_size= embedding_size)
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        Features= self.TCN(x)
        return self.Classifier(Features).squeeze(dim=2)

if __name__=='__main__':
    model = TCN_Model(num_levels=2, num_hidden=200,
                                 embedding_size=2048,
                                 kernel_size=3, dropout=0.2)
    x=torch.rand(16,8,2048)
    model(x)