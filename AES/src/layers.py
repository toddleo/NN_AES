import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False


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
        self.bilstm = nn.LSTM(batch_first=True, input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, bidirectional=True)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input):
        # embedded = self.embedding(input).view(1, 1, -1)
        # output = embedded
        h_0 = Variable(torch.zeros(2, len(input), self.hidden_size), requires_grad=False)
        c_0 = Variable(torch.zeros(2, len(input), self.hidden_size), requires_grad=False)
        h_0 = h_0.cuda() if use_cuda else h_0
        c_0 = c_0.cuda() if use_cuda else c_0

        outputs, (h_n, c_n) = self.bilstm(input, (h_0, c_0))
        outputs = self.dropout(outputs)
        return outputs


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
        self.attn = nn.Linear(self.hidden_size, self.output_size)
        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.dropout = nn.Dropout(self.dropout_p)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, mask):
        # embedded = self.embedding(input).view(1, 1, -1)
        # embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(input))
        # attn_weights.data.masked_fill_(mask, -float('inf'))




        _sums = attn_weights.sum(-1).unsqueeze(2).expand_as(attn_weights)  # sums per row
        attn_weights = attn_weights.div(_sums)
        attn_weights = attn_weights * mask

        return attn_weights


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
        logits = logits
        return logits
