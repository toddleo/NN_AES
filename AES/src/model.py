import torch.nn as nn
import torch
import torch.nn.utils as U
import numpy as np
from layers import Embedding, Modeling, Attn, Output
from torch.autograd import Variable
from torch import LongTensor, FloatTensor

use_cuda = torch.cuda.is_available()

class AESModel (nn.Module):
    def __init__(self, config):
        super(AESModel, self).__init__()
        self.config = config
        self.e0 = Embedding(config.embedding_size, config.hidden_size, config)
        self.m0 = Modeling(config.hidden_size, config.hidden_size, config)
        self.a0 = Attn(2 * config.hidden_size, 2 * config.hidden_size, config.max_length, config, dropout_p=config.dropout)
        self.m1 = Modeling(4 * config.hidden_size, config.hidden_size, config)
        # self.m2 = Modeling(config.hidden_size, config.hidden_size, config)
        self.o0 = Output(2 * config.hidden_size * config.max_length, config)

    def forward(self, input):

        e0_o = self.e0(input)

        m0_o = self.m0(e0_o)

        a0_o = self.a0(m0_o)

        concat = torch.cat([m0_o, m0_o * a0_o], 2)

        m1_o = self.m1(concat)
        o0_o = self.o0(m1_o.contiguous().view(self.config.batch_size, -1))

        return o0_o