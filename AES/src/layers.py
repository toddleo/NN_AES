import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

class Padding(nn.Module):
    def __init__(self, input_size, hidden_size, config, n_layers=1):
        super(Padding, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.config = config

        self.padding = nn.ConstantPad1d(input_size, hidden_size)

    def forward(self, input):
        output = self.embedding(input).view(1, 1, -1)
        return output

class Embedding(nn.Module):
    def __init__(self, input_size, hidden_size, config, n_layers=1):
        super(Embedding, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.config = config

        self.embedding = nn.Embedding(input_size, hidden_size)

    def forward(self, input):
        output = self.embedding(input)
        return output


class Modeling(nn.Module):
    def __init__(self, input_size, hidden_size, config, n_layers=1, dropout_p=0.2):
        super(Modeling, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.config = config
        self.dropout_p = dropout_p

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.bilstm = nn.LSTM(batch_first=True, input_size=input_size, hidden_size=hidden_size, num_layers=n_layers)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input):
        # embedded = self.embedding(input).view(1, 1, -1)
        # output = embedded
        h_0 = Variable(torch.zeros(1, len(input), self.hidden_size), requires_grad=False)
        c_0 = Variable(torch.zeros(1, len(input), self.hidden_size), requires_grad=False)
        h_0 = h_0.cuda() if use_cuda else h_0
        c_0 = c_0.cuda() if use_cuda else c_0

        outputs, (h_n, c_n) = self.bilstm(input, (h_0, c_0))
        outputs = self.dropout(outputs)
        return outputs


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mask):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.mask = mask


    def get_mask(self):
        pass

    def forward(self, inputs):

        if isinstance(inputs, PackedSequence):
            # unpack output
            inputs, lengths = pad_packed_sequence(inputs,
                                                  batch_first=self.batch_first)
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)
                            # (batch_size, hidden_size, 1)
                            )

        attentions = F.softmax(F.relu(weights.squeeze()))

        # create mask based on the sentence lengths
        mask = Variable(torch.ones(attentions.size())).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).expand_as(attentions)  # sums per row
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions

class Attn(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, config, n_layers=1, dropout_p=0.2):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.config = config

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, max_length)
        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.dropout = nn.Dropout(self.dropout_p)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, mask):
        # embedded = self.embedding(input).view(1, 1, -1)
        # embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(input))
        # attn_weights.data.masked_fill_(mask, -float('inf'))
        attn_out = torch.bmm(attn_weights, input)

        return attn_out, attn_weights


class Output(nn.Module):
    def __init__(self, hidden_size, config, output_size=4, n_layers=1, dropout_p=0.2):
        super(Output, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.config = config

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.dropout = nn.Dropout(self.dropout_p)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        # embedded = self.embedding(input).view(1, 1, -1)
        # embedded = self.dropout(embedded)
        logits = self.linear(input)

        return logits
