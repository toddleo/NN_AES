from collections import defaultdict
import pandas as pd

import utils as u
import numpy as np
import torch
import time
import pickle
import torch.optim as O
import torch.nn as nn
from torch.autograd import Variable
from torch import LongTensor, FloatTensor, ByteTensor
import torch.nn.functional as F
import sklearn.metrics as sklm


from model import AESModel
from baseline_model import BaseLineModel
from ConfigFile import Configuration
from nltk import word_tokenize, sent_tokenize
from nltk import stem
from functools import reduce

mainPath = '../data/'
max_length_sent = 0
max_num_sent = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using Cuda')

def get_GLOVE_word2vec(glove_path = '../data/', word_emb_size = 50):
    glove_file = "{}glove.6B.{}d.txt".format(glove_path, word_emb_size)
    # total = int(4e5) #6B
    word2vec_dict = {}
    with open(glove_file, 'r') as gf:
        for line in gf:
            emb = line.lstrip().rstrip().split(" ")
            word = emb[0]
            vector = list(map(float, emb[1:]))
            if word in w2i:
                word2vec_dict[word] = vector
            elif word.capitalize() in w2i:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in w2i:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in w2i:
                word2vec_dict[word.upper()] = vector

    return word2vec_dict

def readData(file='MVP_ALL.csv'):
    global max_num_sent, max_length_sent
    stemmer = stem.porter.PorterStemmer()
    df = pd.read_csv(mainPath + file, encoding='latin-1')
    for index, row in df.iterrows():
        # essay = [line for line in row['text'].split('\n')]
        xyz = 'this is an apple.\nThis is another apple. and I am the king of the world\n\ntow lines.'
        sents = row['text'].split('\n')
        sents = filter(lambda x: len(x.strip()) > 0, sents)
        sents = map(lambda x: sent_tokenize(x), sents)
        sents = reduce(lambda x, y: x + y, sents)
        max_num_sent = max(max_num_sent, len(sents))

        essay = [[w2i[stemmer.stem(x)] for x in word_tokenize(u.normalizeString(sent))] for sent in sents]
        essay = [sent for sent in essay if len(sent) > 0]

        max_length_sent = max(max_length_sent, len(max(essay, key=len)))

        mask = [np.ones(len(sent), dtype=np.int) for sent in essay]
        num_sent = len(sents)
        sent_lengths = [len(sent) for sent in essay]
        yield [essay, row['score'], row['text'], mask, num_sent, sent_lengths]
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
    max_num_sent = 0
    max_length_sent = 0
    for ins in instances:
        max_num_sent = max(max_num_sent, len(ins[0]))
        max_length_sent = max(max_length_sent, len(max(ins[0], key=len)))
    config.max_num_sent = max_num_sent
    config.max_length_sent = max_length_sent

    data = [[np.pad(sent, (0, config.max_length_sent - len(sent)), 'constant', constant_values=0) for sent in essay[0]] for essay in instances]
    data = [np.pad(essay, ((0, config.max_num_sent - len(essay)), (0, 0)), 'constant', constant_values=0) for essay in data]

    # data = np.asarray(list(instances[:,0]), dtype=np.int)
    data = Variable(LongTensor(np.asarray(data, dtype=np.int)))
    data = data.cuda() if use_cuda else data

    num_sent = [essay[4] for essay in instances]
    sent_lengths = [essay[5] for essay in instances]
    sent_lengths = reduce(lambda x, y: x + y, sent_lengths)

    num_sent = np.array(num_sent)
    sent_lengths = np.array(sent_lengths)

    mask = [[np.pad(sent, (0, config.max_length_sent - len(sent)), 'constant', constant_values=0) for sent in essay[3]]
            for essay in instances]
    mask = [np.pad(essay, ((0, config.max_num_sent - len(essay)), (0, 0)), 'constant', constant_values=0) for essay in
            mask]
    mask = Variable(FloatTensor(np.asarray(mask, dtype=np.float)))
    mask = mask.cuda() if use_cuda else mask

    label = np.asarray([ins[1] - 1 for ins in instances])
    label = Variable(LongTensor(label))

    inp = torch.unsqueeze(label, 1)

    label = label.cuda() if use_cuda else label



    # inp = inp.cuda() if use_cuda else inp

    one_hot_label = torch.FloatTensor(len(instances), config.output_size).zero_()
    # one_hot_label = one_hot_label.cuda() if use_cuda else one_hot_label
    one_hot_label.scatter_(1, inp.data, 1)
    one_hot_label = Variable(FloatTensor(one_hot_label))
    one_hot_label = one_hot_label.cuda() if use_cuda else one_hot_label
    #print(label)
    #print(one_hot)

    return data, mask, label, num_sent, sent_lengths, one_hot_label

if __name__ == '__main__':
    config = Configuration()
    w2i = defaultdict(lambda: len(w2i))
    eos = '<eos>'
    w2i[eos]
    data = list(readData(file='ASAP4.csv'))

    config.max_length_sent = max_length_sent
    config.max_num_sent = max_num_sent

    print('Total:' + str(len(data)))
    print('Max Num Sent: ' + str(config.max_num_sent))
    print('Max Sent Length: ' + str(config.max_length_sent))

    unk_src = w2i["<unk>"]
    w2i = defaultdict(lambda: unk_src, w2i)
    config.vocab_size = len(w2i)
    i2w = {i: w for w, i in w2i.items()}

    # data = Variablelize(data)
    if config.needCreateEM:
        word2vec_dict = get_GLOVE_word2vec()
        widx2vec_dict = {w2i[word]: vec for word, vec in word2vec_dict.items() if word in w2i}
        emb_mat = np.array([widx2vec_dict[wid] if wid in widx2vec_dict
                            else np.random.multivariate_normal(np.zeros(config.embedding_output), np.eye(config.embedding_output))
                            for wid in range(config.vocab_size+1)])
        config.emb_mat = emb_mat
        with open('../data/emb_mat.pkl', 'wb') as f:
            pickle.dump(emb_mat, f)
    else:
        with open('../data/emb_mat.pkl', 'rb') as f:
            emb_mat = pickle.load(f)



    trainset, devtest = split(data, test_size=0.4)

    devset, testset = split(devtest, test_size=0.5)

    model = BaseLineModel(config)
    # model = AESModel(config)
    if use_cuda:
        model.cuda()

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # optimizer = O.Adadelta(model.parameters(), config.init_lr)
    # optimizer = O.Adam(model.parameters())
    # optimizer = O.SGD(model.parameters(), lr = 0.01)
    optimizer = O.RMSprop(model.parameters(), lr=0.001, momentum=0.9)

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

            data, mask, label, num_sent, sent_lengths, one_hot_label = variablelize(instances)

            output = model.forward(data, mask, num_sent, sent_lengths)

            # output = F.sigmoid(output)
            # values, predict = torch.max(output, 1)
            # output = predict.float()

            loss = criterion(F.sigmoid(output), one_hot_label)
            # loss = criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optimizer.step()

            total_loss += loss.data[0] * len(instances)
            numOfBatch += 1
            numOfSamples += len(instances)
            if numOfBatch % 50 == 0:
                end = time.time()
                total_dev_loss = 0
                predicts = []
                model.eval()
                for tid in range(0, len(devset), config.test_batch_size):
                    dev_instances = testset[tid:tid + config.test_batch_size]
                    data, mask, label, num_sent, sent_lengths, one_hot_label = variablelize(dev_instances)

                    output = model.forward(data, mask, num_sent, sent_lengths)

                    # values, predict = torch.max(F.softmax(output), 1)
                    # predict = predict.cpu().data.numpy()
                    # predicts.extend(predict)


                    dev_loss = criterion(F.sigmoid(output), one_hot_label)
                    # dev_loss = criterion(output, label)
                    total_dev_loss += dev_loss.data[0] * len(dev_instances)

                print (str(epoch) + " , " + str(numOfSamples) + ' / ' + str(len(trainset)) + " , Current loss : " + str(
                    total_loss / numOfSamples) + ", test loss: " + str(total_dev_loss / len(devset)) + ", run time = " + str(end - start))
                start = time.time()

        predicts = []
        # attentions = []
        model.eval()
        for tid in range(0, len(testset), config.test_batch_size):
            test_instances = testset[tid:tid + config.test_batch_size]

            data, mask, label, num_sent, sent_lengths, one_hot_label = variablelize(test_instances)

            output = model.forward(data, mask, num_sent, sent_lengths)

            values, predict = torch.max(F.sigmoid(output), 1)

            # values, predict = torch.max(F.softmax(output), 1)
            predict = predict.cpu().data.numpy()
            predicts.extend(predict)
            # attentions.extend(test_attention)

        qwkappa = sklm.cohen_kappa_score([ins[1] - 1 for ins in testset], predicts, labels=[0, 1, 2, 3],
                                         weights='quadratic')
        torch.save(model.state_dict(), '../models/' + str(epoch) + '.pkl')
        print("kappa = " + str(qwkappa))