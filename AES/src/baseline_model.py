import torch.nn as nn
import torch
import torch.nn.utils as U
import numpy as np
from baseline_layers import Embedding, Conv, Modeling, Attn, Output
import torch.nn.functional as F
from torch.autograd import Variable
from torch import LongTensor, FloatTensor
from torch.nn.utils.rnn import pack_padded_sequence

use_cuda = torch.cuda.is_available()

class BaseLineModel (nn.Module):
    def __init__(self, config):
        super(BaseLineModel, self).__init__()
        self.config = config
        self.e0 = Embedding(config.vocab_size, config.embedding_output, config)
        self.c0 = Conv(1, config.num_of_filters, (config.window_size, config.embedding_output), config)
        # self.c0 = Conv(config.embedding_output, config.num_of_filters, config.window_size, config)

        # self.m0 = Modeling(config.embedding_output, config.hidden_size, config)
        self.a0 = Attn(config.num_of_filters, config.num_of_filters, config)
        self.m0 = Modeling(config.num_of_filters, config.hidden_size, config)
        self.a1 = Attn(config.hidden_size, config.hidden_size, config)
        self.o0 = Output(config.hidden_size, config, config.output_size)

    def forward(self, input, mask, num_sent, sent_lengths, outputAttn=False):
        # mask = mask.unsqueeze(3).repeat(1, 1, 1, self.config.embedding_output)
        # print(input.data.size())
        e0_o = self.e0(input)
        # e0_o = e0_o.transpose(1, 2)
        e0_o = e0_o.unsqueeze(1)
        c0_o = self.c0(e0_o)
        c0_o = c0_o.transpose(1, 2)
        c0_o = F.tanh(c0_o).squeeze(3)
        # c0_o = c0_o.transpose(1, 2)
        a0_o = self.a0(c0_o)

        s0_o = a0_o * c0_o
        s0_o = s0_o.sum(1)
        # s0_o = s0_o.view(len(input), self.config.max_num_sent, -1)

        # a0_o = self.a0(e0_o, mask)
        # print(a0_o.data.size())
        t0_o = torch.zeros(len(num_sent), self.config.max_num_sent, self.config.num_of_filters)

        count = 0
        for i in range(len(num_sent)):
            for j in range(num_sent[i]):

                t0_o[i][j] = s0_o.data[count]
                count = count + 1
        t0_o = Variable(t0_o)
        t0_o = t0_o.cuda() if use_cuda else t0_o

        m0_o = self.m0(t0_o)
        a1_o = self.a1(m0_o, mask.select(2, 0).unsqueeze(2).repeat(1, 1, self.config.hidden_size))

        s1_o = a1_o * m0_o
        s1_o = s1_o.sum(1)

        o0_o = self.o0(s1_o)
        # o0_o = self.o0(concat.contiguous().view(len(input), -1))
        # print(o0_o.data.size())

        


        if outputAttn:
            return o0_o, a0_o
        else:
            return o0_o

