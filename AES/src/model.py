import torch.nn as nn
import torch
import torch.nn.utils as U
import numpy as np
from layers import Embedding, Modeling, Attn, Output
from torch.autograd import Variable
from torch import LongTensor, FloatTensor
from torch.nn.utils.rnn import pack_padded_sequence

use_cuda = torch.cuda.is_available()

class AESModel (nn.Module):
    def __init__(self, config):
        super(AESModel, self).__init__()
        self.config = config
        self.e0 = Embedding(config.vocab_size, config.embedding_output, config)
        self.m0 = Modeling(config.embedding_output, config.hidden_size, config)
        self.a0 = Attn(2 * config.hidden_size, 2 * config.hidden_size, config.max_length_sent, config, dropout_p=config.dropout)
        self.m1 = Modeling(4 * config.hidden_size, config.hidden_size, config)
        self.a1 = Attn(2 * config.hidden_size, 2 * config.hidden_size, config.max_length_sent, config,
                       dropout_p=config.dropout)
        self.m2 = Modeling(4 * config.hidden_size, config.hidden_size, config)
        # self.m2 = Modeling(config.hidden_size, config.hidden_size, config)
        self.o0 = Output(2 * config.hidden_size * config.max_length_sent * config.max_num_sent, config)
        # self.o0 = Output(2 * config.hidden_size, config)

    def sort_batch(self, data, seq_len):
        batch_size = data.size(0)
        sorted_seq_len, sorted_idx = seq_len.sort()
        reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()
        sorted_seq_len = sorted_seq_len[reverse_idx]
        sorted_data = data[sorted_idx][reverse_idx]
        return sorted_data, sorted_seq_len

    def desort_batch(self, sorted_data, sorted_seq_len):
        unsorted = sorted_data.new(*sorted_data.size())
        unsorted.scatter_(0, sorted_seq_len, sorted_data)

    def forward(self, input, mask, num_sent, sent_lengths, outputAttn=False):
        # mask = mask.unsqueeze(3).repeat(1, 1, 1, 2 * self.config.hidden_size)
        ns = Variable(torch.LongTensor(num_sent))
        sorted_data, sorted_seq_len = self.sort_batch(input, ns)

        packed = pack_padded_sequence(sorted_data, list(sorted_seq_len.data), True)
        sl = Variable(torch.LongTensor(sent_lengths))
        sorted_data, sorted_seq_len = self.sort_batch(packed.data, sl)
        num_sent_ind = np.argsort(-num_sent)
        num_sent_ord = num_sent[num_sent_ind]

        sent_lengths_ind = np.argsort(-sent_lengths)
        sent_lengths_ord = sent_lengths[sent_lengths_ind]

        e0_o = self.e0(input.view(len(input), -1))
        # print(e0_o.data.size())



        e0_o = e0_o.view(len(input), self.config.max_num_sent, self.config.max_length_sent, -1)
        m0_o = Variable(
                torch.zeros((len(input), self.config.max_num_sent, self.config.max_length_sent, 2 * self.config.hidden_size)).type(torch.FloatTensor))
        m0_o = m0_o.cuda() if use_cuda else m0_o

        for ei in range(len(e0_o)):
            m0_o[ei] = self.m0(e0_o[ei])
        # print(m0_o.data.size())

        m0_o = m0_o.view(len(input), self.config.max_num_sent * self.config.max_length_sent, -1)
        # mask = mask.view(len(input), self.config.max_num_sent * self.config.max_length_sent, -1)
        a0_o = self.a0(m0_o, mask.view(len(input), self.config.max_num_sent * self.config.max_length_sent, -1))
        # a0_o = self.a0(e0_o, mask)
        # print(a0_o.data.size())

        a0_o = a0_o.view(len(input), self.config.max_num_sent * self.config.max_length_sent, -1)
        concat = torch.cat([a0_o, a0_o*m0_o], 2)
        concat = concat.view(len(input), self.config.max_num_sent, self.config.max_length_sent, -1)
        # print(concat.data.size())

        m1_o = Variable(
            torch.zeros(
                (len(input), self.config.max_num_sent, self.config.max_length_sent, 2 * self.config.hidden_size)).type(
                torch.FloatTensor))
        m1_o = m1_o.cuda() if use_cuda else m1_o
        for ci in range(len(concat)):
            # print(e0_o[0][0][0][0])
            m1_o[ei] = self.m1(concat[ci])
            # print(e0_o[0][0][0][0])
        # print(m1_o.data.size())

        # m1_o = m1_o * mask
        # , mask.view(len(input), -1)
        o0_o = self.o0(m1_o.contiguous().view(len(input), -1))
        # o0_o = self.o0(concat.contiguous().view(len(input), -1))
        # print(o0_o.data.size())
        if outputAttn:
            return o0_o, a0_o
        else:
            return o0_o

