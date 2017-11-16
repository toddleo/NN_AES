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
        if index >= 20:
            break


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
            optimizer.zero_grad()

            instances = trainset[sid:sid + config.batch_size]
            # print(instances[0][2])
            data = np.asarray(
                [np.pad(ins[0], (0, config.max_length - len(ins[0])), 'constant', constant_values=0) for ins in instances])

            data = Variable(LongTensor(data))
            data = data.cuda() if use_cuda else data

            mask = [np.pad(ins[3], (0, config.max_length - len(ins[0])), 'constant', constant_values=0) for ins in instances]
            mask = Variable(FloatTensor(mask))
            mask = mask.cuda() if use_cuda else mask

            label = np.asarray([ins[1]-1 for ins in instances])
            label = Variable(LongTensor(label))
            label = label.cuda() if use_cuda else label

            output = model.forward(data, mask)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]
            numOfBatch += 1
            numOfSamples += len(instances)
            if numOfBatch % 10 == 0:
                end = time.time()
                total_test_loss = 0
                predicts = []
                for tid in range(0, len(devset), config.test_batch_size):
                    dev_instances = testset[tid:tid + config.test_batch_size]
                    dev_data = np.asarray(
                        [np.pad(ins[0], (0, config.max_length - len(ins[0])), 'constant', constant_values=0) for ins in
                         dev_instances])
                    dev_data = Variable(LongTensor(dev_data))
                    dev_data = dev_data.cuda() if use_cuda else dev_data

                    dev_mask = np.asarray(
                        [np.pad(ins[3], (0, config.max_length - len(ins[0])), 'constant', constant_values=0) for ins in
                         dev_instances])
                    dev_mask = Variable(LongTensor(dev_mask))
                    dev_mask = dev_mask.cuda() if use_cuda else dev_mask

                    true_label = np.asarray([ins[1] - 1 for ins in dev_instances])
                    true_label = Variable(LongTensor(true_label))
                    true_label = true_label.cuda() if use_cuda else true_label

                    test_out = model.forward(dev_data, dev_mask)
                    values, predict = torch.max(F.softmax(test_out), 1)
                    predict = predict.cpu().data.numpy()
                    predicts.extend(predict)
                    test_loss = criterion(test_out, true_label)
                    total_test_loss += test_loss.data[0]

                print (str(epoch) + " , " + str(numOfSamples) + ' / ' + str(len(trainset)) + " , Current loss : " + str(
                    total_loss / numOfSamples) + ", test loss: " + str(total_test_loss / len(dev_data)) + ", run time = " + str(end - start))
                start = time.time()

        predicts = []
        for tid in range(0, len(testset), config.test_batch_size):
            test_instances = testset[tid:tid + config.test_batch_size]
            test_data = np.asarray(
                [np.pad(ins[0], (0, config.max_length - len(ins[0])), 'constant', constant_values=0) for ins in
                 test_instances])
            test_data = Variable(LongTensor(test_data))
            test_data = test_data.cuda() if use_cuda else test_data

            test_mask = np.asarray(
                [np.pad(ins[3], (0, config.max_length - len(ins[0])), 'constant', constant_values=0) for ins in
                 test_instances])
            test_mask = Variable(LongTensor(test_mask))
            test_mask = test_mask.cuda() if use_cuda else test_mask

            true_label = np.asarray([ins[1] - 1 for ins in test_instances])
            true_label = Variable(LongTensor(true_label))
            true_label = true_label.cuda() if use_cuda else true_label

            test_out = model.forward(test_data, test_mask)
            values, predict = torch.max(F.softmax(test_out), 1)
            predict = predict.cpu().data.numpy()
            predicts.extend(predict)
            test_loss = criterion(test_out, true_label)
            total_test_loss = test_loss.data[0]

        qwkappa = sklm.cohen_kappa_score([ins[1] - 1 for ins in testset], predicts, labels=[0, 1, 2, 3],
                                         weights='quadratic')
        print ("kappa = " + str(qwkappa))