from collections import defaultdict
import pandas as pd
import utils as u
import numpy as np
import torch
import time
import os
import torch.optim as O
import torch.nn as nn
from torch.autograd import Variable
from torch import LongTensor, FloatTensor
import torch.nn.functional as F
import sklearn.metrics as sklm


from model import AESModel
from ConfigFile import Configuration
from nltk import word_tokenize
from nltk import stem

mainPath = '../data/'
max_length = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    print ('Using Cuda')

def readData(file='MVP_ALL.csv'):
    global max_length
    stemmer = stem.porter.PorterStemmer()
    df = pd.read_csv(mainPath + file)
    for index, row in df.iterrows():

        essay = [w2i[stemmer.stem(x)] for x in word_tokenize(u.normalizeString(row['text']))]
        max_length = max(max_length, len(essay))
        yield (essay, row['score'], row['text'], np.ones(len(essay), dtype=np.int))
        # if index >= 20:
        #     break


def split(data, test_size=0.2, shuffle=True, random_seed=42):
    num_train = len(data)
    indices = list(range(num_train))
    split = int(np.floor(test_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_set = [data[i] for i in train_idx]
    test_set = [data[i] for i in test_idx]
    return train_set, test_set

def Variablelize(data):
    for instance in data:
        instance = list(instance)
        instance[3] = Variable(torch.LongTensor(instance[0]))
    return data

def printAttention(true_labels, predict_labels, attention, data):
    for i in range(len(true_labels)):
        if true_labels[i] == predict_labels[i] and predict_labels[i] == 1:
            essay = data[i]
            attn = attention[i]
            max_attn = attn.max(1)[0]
            for j in range(len(essay)):
                if essay[j].data[0] != 0:
                    print(i2w[essay[j].data[0]] + '\t' + str(max_attn[j].data[0]))


if __name__ == '__main__':
    config = Configuration()
    w2i = defaultdict(lambda: len(w2i))
    eos = '<eos>'
    w2i[eos]
    data = list(readData())
    print(len(data))
    config.max_length = max_length
    print (config.max_length)
    unk_src = w2i["<unk>"]
    w2i = defaultdict(lambda: unk_src, w2i)
    config.embedding_size = len(w2i)
    i2w = {i: w for w, i in w2i.items()}

    # data = Variablelize(data)

    trainset, devtest = split(data, test_size=0.4)

    devset, testset = split(devtest, test_size=0.5)


    model = AESModel(config)
    if use_cuda:
        model.cuda()
    if os.path.isfile('../models/50_20.pkl'):
        model.load_state_dict(torch.load('../models/50_20.pkl'))
        print('Loading model...')
    model.eval()


    numOfBatch = 0
    predicts = []
    for sid in range(0, len(testset), config.batch_size):
        instances = testset[sid:sid + config.batch_size]
        # print(instances[0][2])
        data = np.asarray(
            [np.pad(ins[0], (0, config.max_length - len(ins[0])), 'constant', constant_values=0) for ins in instances])

        data = Variable(LongTensor(data))
        data = data.cuda() if use_cuda else data

        mask = [np.pad(ins[3], (0, config.max_length - len(ins[0])), 'constant', constant_values=0) for ins in
                instances]
        mask = Variable(FloatTensor(mask))
        mask = mask.cuda() if use_cuda else mask

        label = np.asarray([ins[1] - 1 for ins in instances])

        output, attention = model.forward(data, mask)

        values, predict = torch.max(F.softmax(output), 1)
        predict = predict.cpu().data.numpy()



        printAttention(label, predict, attention, data)

        predicts.extend(predict)
        # attentions.extend(test_attention)

    qwkappa = sklm.cohen_kappa_score([ins[1] - 1 for ins in testset], predicts, labels=[0, 1, 2, 3],
                                     weights='quadratic')
    print ("kappa = " + str(qwkappa))