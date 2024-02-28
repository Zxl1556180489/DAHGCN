from torch import nn
from models import DAHGCN_conv
import torch.nn.functional as F


class DAHGCN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(DAHGCN, self).__init__()
        self.dropout = dropout
        self.hgc1 = DAHGCN_conv(in_ch, n_hid)
        self.hgc2 = DAHGCN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x
