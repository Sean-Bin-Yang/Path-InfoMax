import torch
import math
import torch.nn as nn
from layers import Discriminator
from utils import process
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 n_layers=2, dropout=0.5):
        super(EncoderLSTM, self).__init__()

        ##can be initialized by results from graph embedding methods, e.g. node2vec.
        self.embedding= nn.Embedding(8894, 128)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout)

    def forward(self, path, hidden=None):

        PathEmbedOri= self.embedding(path)
        PathEmbed = PathEmbedOri.transpose(1,0)

        outputs, hidden = self.lstm(PathEmbed, hidden)
        outputs = outputs.transpose(0,1)

        return outputs, hidden, PathEmbedOri

class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            return torch.sum(seq * msk, 1) / torch.sum(msk,1) 

class PIM(nn.Module):
    def __init__(self, n_in, n_h, input_size=128, hidden_size= 128, n_layers=1, dropout=0.5):

        super(PIM, self).__init__()

        self.encoder_path = EncoderLSTM(input_size=input_size, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
        
        
        self.disc_node = Discriminator(n_in, n_h)
        self.disc_path = Discriminator(n_in, n_h)
        
        self.read = self.read = Readout()

        ##can be initialized by results from graph embedding methods, e.g. node2vec.
        self.embeddingn= nn.Embedding(8894,128)


    def forward(self, Path1, Path1Mask, Path2, Path2Mask, Path3, Path3Mask, Pos1, Pos1Mask, Neg1, Neg1Mask, Neg2, Neg2Mask):


        ###path 
        Path1Out, _, Path1Ori = self.encoder_path(Path1)
        Path1Embed = torch.unsqueeze(self.read(Path1Out, Path1Mask),1)
        Path1OriP = torch.unsqueeze(self.read(Path1Ori, Path1Mask),1)

        Path2Out, _, _ = self.encoder_path(Path2)
        Path2Embed = torch.unsqueeze(self.read(Path2Out, Path2Mask),1) 
    

        Path3Out, _, _ = self.encoder_path(Path3)
        Path3Embed = torch.unsqueeze(self.read(Path3Out, Path3Mask),1)

        ###Node embedding
        NEmbedPos = self.embeddingn(Pos1)
        NEmbedNeg1 = self.embeddingn(Neg1)
        NEmbedNeg2= self.embeddingn(Neg2)

        ##node_discriminator
        logits1 = self.disc_node(Path1Embed, Path1Ori, NEmbedPos*Pos1Mask, NEmbedNeg1*Neg1Mask, NEmbedNeg2*Neg2Mask)  
        
        ##path_discriminator
        logits2 = self.disc_path(Path1Embed, Path1OriP, Path1OriP, Path2Embed, Path3Embed)
        
        
        return logits1, logits2

    def embed(self, Path1, Path1Mask):
        
        #path embedding
        Path1Out, _, _ = self.encoder_path(Path1)
        Path1Embed = torch.unsqueeze(self.read(Path1Out, Path1Mask),1)

        return Path1Embed.detach()