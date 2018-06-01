#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import random
import itertools

import numpy as np

import base
from base import load_pension_lottery_csv

INPUT_PATH = '../../../datasets/pension_lottery_data.csv'

def load_pension_lottery_fcn_datasets(n_data=1, shuffle=False, ratio=[0.9, None, 0.1]):
    input = load_pension_lottery_csv(INPUT_PATH)
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

    train = base.Dataset(data=data[:n_train], target=target[:n_train])

    if ratio[1]:
        n_test = int(n_data * ratio[2])
        validation = base.Dataset(data=data[n_train:n_data-n_test], target=target[n_train:n_data-n_test])
        test = base.Dataset(data=data[n_data-n_test:], target=target[n_data-n_test:])
    else:
        validation = None
        test = base.Dataset(data=data[n_train:], target=target[n_train:])

    return base.Datasets(train=train, validation=validation, test=test)

if __name__ == '__main__':
    datasets = load_pension_lottery_fcn_datasets(n_data=2, shuffle=True)

    print(datasets.test.target[0])
