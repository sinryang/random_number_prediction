#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import random
import urllib
import itertools
import collections

import numpy as np

INPUT_PATH = '../datasets/pension_lottery_data.csv'
url = 'https://raw.githubusercontent.com/sinryang/random_number_prediction/master/datasets/pension_lottery_data.csv'

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def load_pension_lottery_data_from_file(filename):
    with open(filename) as fin:
        inputs = list(csv.reader(fin))[1:]
        data = []

        for row in inputs:
            new_row = []

            for i in range(len(row)):
                new_row += [float(e)/10 for e in row[i]]

            data.append(new_row)

    return data

def load_pension_lottery_data_from_url(url):
    fin = urllib.urlopen(url)
    inputs = list(csv.reader(fin))[1:]
    data = []

    for row in inputs:
        new_row = []

        for i in range(len(row)):
            new_row += [float(e)/10 for e in row[i]]

        data.append(new_row)

    return data

def load_pension_lottery_fcn_datasets(n_data=1, shuffle=False, ratio=[0.9, None, 0.1]):
    if os.path.exists(INPUT_PATH):
        print('load data from file...')
        input = load_pension_lottery_data_from_file(INPUT_PATH)
    else:
        print('load data from url...')
        input = load_pension_lottery_data_from_url(url)

    n_feature = len(input[0])

    data , target = [], []

    for i in range(n_data, len(input)):
        data.append(sum(input[i-n_data:i+1], []))

    if shuffle:
        random.shuffle(data)

    for i in range(len(data)):
        target.append(np.array(data[i][-n_feature:], dtype=np.float32))
        data[i] = np.array(data[i][:n_feature*n_data], dtype=np.float32)

    data = np.array(data)
    target = np.array(target)

    n_data = len(data)
    n_train = int(n_data * ratio[0])

    train = Dataset(data=data[:n_train], target=target[:n_train])

    if ratio[1]:
        n_test = int(n_data * ratio[2])
        validation = Dataset(data=data[n_train:n_data-n_test], target=target[n_train:n_data-n_test])
        test = Dataset(data=data[n_data-n_test:], target=target[n_data-n_test:])
    else:
        validation = None
        test = Dataset(data=data[n_train:], target=target[n_train:])

    return Datasets(train=train, validation=validation, test=test)

if __name__ == '__main__':
    datasets = load_pension_lottery_fcn_datasets(n_data=2, shuffle=True)

    print(datasets.test.target[0])
