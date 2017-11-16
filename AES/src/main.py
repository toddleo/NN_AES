from collections import defaultdict
import pandas as pd
import utils as u
import numpy as np
import torch
import time
import torch.optim as O
import torch.nn as nn
from torch.autograd import Variable
from torch import LongTensor, FloatTensor

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
        yield (essay, row['score'], row['text'])
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


    trainset, testset = split(data)



    model = AESModel(config)
    if use_cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = O.Adadelta(model.parameters(), config.init_lr)

    for epoch in range(0, config.epochs):
        total_loss = 0
        # need to implement BATCH
        numOfSamples = 0
        numOfBatch = 0
        start = time.time()
        # for instance in train:
        print("Start Training:" + str(epoch))
        for sid in range(0, len(trainset), config.batch_size):
            optimizer.zero_grad()

            instances = trainset[sid:sid + config.batch_size]
            # print(instances[0][2])
            data = np.asarray(
                [np.pad(ins[0], (0, config.max_length - len(ins[0])), 'constant', constant_values=0) for ins in instances])

            data = Variable(LongTensor(data))
            data = data.cuda() if use_cuda else data

            label = np.asarray([ins[1]-1 for ins in instances])
            label = Variable(LongTensor(label))
            label = label.cuda() if use_cuda else label

            output = model.forward(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]
            numOfBatch += 1
            numOfSamples += len(instances)
            if numOfBatch % 10 == 0:
                print("testing")
                end = time.time()
                total_test_loss = 0
                for tid in range(0, len(testset), config.test_batch_size):
                    test_instances = trainset[sid:sid + config.test_batch_size]
                    test_data = np.asarray(
                        [np.pad(ins[0], (0, config.max_length - len(ins[0])), 'constant', constant_values=0) for ins in
                         test_instances])
                    test_data = Variable(LongTensor(test_data))
                    test_data = test_data.cuda() if use_cuda else test_data

                    true_label = np.asarray([ins[1] - 1 for ins in testset])
                    true_label = Variable(LongTensor(true_label))
                    true_label = true_label.cuda() if use_cuda else true_label

                    test_out = model.forward(test_data)
                    test_loss = criterion(test_out, true_label)
                    total_test_loss = test_loss.data[0]
                print (str(epoch) + " , " + str(numOfSamples) + ' / ' + str(len(trainset)) + " , Current loss : " + str(
                    total_loss / numOfSamples) + ", test loss: " + str(total_test_loss / len(test_data)) + ", run time = " + str(end - start))
                start = time.time()

    print ("this is the end")