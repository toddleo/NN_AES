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
        self.e0 = Embedding(config.embedding_size, config.embedding_output, config)
        self.m0 = Modeling(config.embedding_output, config.hidden_size, config)
        self.a0 = Attn(2 * config.hidden_size, 2 * config.hidden_size, config.max_length, config, dropout_p=config.dropout)
        self.m1 = Modeling(4 * config.hidden_size, config.hidden_size, config)
        # self.m2 = Modeling(config.hidden_size, config.hidden_size, config)
        self.o0 = Output(2 * config.hidden_size * config.max_length, config)
        # self.o0 = Output((2 * config.hidden_size + config.max_length) * config.max_length, config)

    def forward(self, input, mask):
        mask = mask.unsqueeze(2).repeat(1, 1, 2 * self.config.hidden_size)

        e0_o = self.e0(input)
        # print (e0_o.data.size())
        m0_o = self.m0(e0_o)
        # print (m0_o.data.size())
        a0_o = self.a0(m0_o, mask)
        # a0_o = self.a0(e0_o, mask)
        # print (a0_o.data.size())
        concat = torch.cat([a0_o, a0_o*m0_o], 2)
        # print (concat.data.size())
        m1_o = self.m1(concat)
        # print (m1_o.data.size())
        # m1_o = m1_o * mask
        o0_o = self.o0(m1_o.contiguous().view(len(input), -1))
        # o0_o = self.o0(concat.contiguous().view(len(input), -1))
        # print (o0_o.data.size())
        return o0_o, a0_o

