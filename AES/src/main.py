from collections import defaultdict
import pandas as pd

import utils as u
import numpy as np
import torch
import time
import torch.optim as O
import torch.nn as nn
from torch.autograd import Variable
from torch import LongTensor, FloatTensor, ByteTensor
import torch.nn.functional as F
import sklearn.metrics as sklm


from model import AESModel
from ConfigFile import Configuration
from nltk import word_tokenize, sent_tokenize
from nltk import stem
from functools import reduce

mainPath = '../data/'
max_length_sent = 0
max_num_sent = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    print ('Using Cuda')


def readData(file='MVP_ALL.csv'):
    global max_num_sent, max_length_sent
    stemmer = stem.porter.PorterStemmer()
    df = pd.read_csv(mainPath + file, encoding='utf-8')
    for index, row in df.iterrows():
        # essay = [line for line in row['text'].split('\n')]
        xyz = 'this is an apple.\nThis is another apple. and I am the king of the world\n\ntow lines.'
        sents = row['text'].split('\n')
        sents = filter(lambda x: len(x) > 0, sents)
        sents = map(lambda x: sent_tokenize(x), sents)
        sents = reduce(lambda x, y: x + y, sents)
        max_num_sent = max(max_num_sent, len(sents))

        essay = [[w2i[stemmer.stem(x)] for x in word_tokenize(u.normalizeString(sent))] for sent in sents]
        max_length_sent = max(max_length_sent, len(max(essay, key=len)))

        mask = [np.ones(len(sent), dtype=np.int) for sent in essay]

        yield (essay, row['score'], row['text'], mask)
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


def variablelize(instances):
    # max_num_sent = 0
    # max_length_sent = 0
    # for ins in instances:
    #     max_num_sent = max(max_num_sent, len(ins[0]))
    #     max_length_sent = max(max_length_sent, len(max(ins[0], key=len)))
    # config.max_num_sent = max_num_sent
    # config.max_length_sent = max_length_sent

    data = [[np.pad(sent, (0, config.max_length_sent - len(sent)), 'constant', constant_values=0) for sent in essay[0]] for essay in instances]
    data = [np.pad(essay, ((0, config.max_num_sent - len(essay)), (0, 0)), 'constant', constant_values=0) for essay in data]

    data = Variable(LongTensor(np.asarray(data, dtype=np.int)))
    data = data.cuda() if use_cuda else data

    mask = [[np.pad(sent, (0, config.max_length_sent - len(sent)), 'constant', constant_values=0) for sent in essay[3]]
            for essay in instances]
    mask = [np.pad(essay, ((0, config.max_num_sent - len(essay)), (0, 0)), 'constant', constant_values=0) for essay in
            mask]
    mask = Variable(FloatTensor(np.asarray(mask, dtype=np.float)))
    mask = mask.cuda() if use_cuda else mask

    label = np.asarray([ins[1] - 1 for ins in instances])
    label = Variable(LongTensor(label))
    label = label.cuda() if use_cuda else label

    return data, mask, label

if __name__ == '__main__':
    config = Configuration()
    w2i = defaultdict(lambda: len(w2i))
    eos = '<eos>'
    w2i[eos]
    data = list(readData())

    config.max_length_sent = max_length_sent
    config.max_num_sent = max_num_sent

    print('Total:' + str(len(data)))
    print('Max Num Sent: ' + str(config.max_num_sent))
    print('Max Sent Length: ' + str(config.max_length_sent))

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

    criterion = nn.CrossEntropyLoss()
    # optimizer = O.Adadelta(model.parameters(), config.init_lr)
    optimizer = O.Adam(model.parameters())
    # optimizer = O.SGD(model.parameters(), lr = 0.01)

    for epoch in range(0, config.epochs):
        total_loss = 0
        # need to implement BATCH
        numOfSamples = 0
        numOfBatch = 0
        start = time.time()
        # for instance in train:
        print("Start Training:" + str(epoch))
        for sid in range(0, len(trainset), config.batch_size):
            model.train()
            optimizer.zero_grad()

            instances = trainset[sid:sid + config.batch_size]
            # print(instances[0][2])

            data, mask, label = variablelize(instances)

            output = model.forward(data, mask)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0] * len(instances)
            numOfBatch += 1
            numOfSamples += len(instances)
            if numOfBatch % 10 == 0:
                end = time.time()
                total_dev_loss = 0
                predicts = []
                model.eval()
                for tid in range(0, len(devset), config.test_batch_size):
                    dev_instances = testset[tid:tid + config.test_batch_size]
                    data, mask, label = variablelize(dev_instances)

                    output = model.forward(data, mask)
                    values, predict = torch.max(F.softmax(output), 1)
                    predict = predict.cpu().data.numpy()
                    predicts.extend(predict)
                    dev_loss = criterion(output, label)
                    total_dev_loss += dev_loss.data[0] * len(dev_instances)

                print (str(epoch) + " , " + str(numOfSamples) + ' / ' + str(len(trainset)) + " , Current loss : " + str(
                    total_loss / numOfSamples) + ", test loss: " + str(total_dev_loss / len(devset)) + ", run time = " + str(end - start))
                start = time.time()

        predicts = []
        # attentions = []
        model.eval()
        for tid in range(0, len(testset), config.test_batch_size):
            test_instances = testset[tid:tid + config.test_batch_size]

            data, mask, label = variablelize(test_instances)

            output = model.forward(data, mask)
            values, predict = torch.max(F.softmax(output), 1)
            predict = predict.cpu().data.numpy()
            predicts.extend(predict)
            # attentions.extend(test_attention)

        qwkappa = sklm.cohen_kappa_score([ins[1] - 1 for ins in testset], predicts, labels=[0, 1, 2, 3],
                                         weights='quadratic')
        torch.save(model.state_dict(), '../models/' + str(epoch) + '.pkl')
        print ("kappa = " + str(qwkappa))