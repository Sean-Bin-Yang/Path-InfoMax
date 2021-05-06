import torch
import numpy as np
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_h1, n_h2):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h1, n_h2, 1)
        self.act = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_c, h_ori, h_p, h_n1, h_n2, s_bias1=None, s_bias2=None):

        ##Positve samples
        h_c = h_c.expand_as(h_ori).contiguous()
        h_p = h_p.expand_as(h_ori).contiguous()

        sc_p1 = torch.squeeze(self.f_k(h_p, h_c), 2)

        ##Negative samples
        h_n1 = h_n1.expand_as(h_ori).contiguous()
        h_n2 = h_n2.expand_as(h_ori).contiguous()


        sc_n1 = torch.squeeze(self.f_k(h_n1, h_c), 2)
        sc_n2 = torch.squeeze(self.f_k(h_n2, h_c), 2)

        logits = torch.cat((sc_p1, sc_n1, sc_n2), 1)

        return logits