#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import collections

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def load_pension_lottery_csv(filename):
    with open(filename) as fin:
        inputs = list(csv.reader(fin))[1:]
        data = []

        for row in inputs:
            new_row = []

            for i in range(len(row)):
                new_row += [float(e)/10 for e in row[i]]

            data.append(new_row)

    return data
