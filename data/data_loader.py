'''
Created on Mar 31, 2019

@author: dulan
'''
import logging
import numpy as np
import pandas as pd
import os
import json

class data_loader(object):

    def __init__(self):
        logging.basicConfig(level=logging.INFO)

    def load_dict(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r') as fp:
                dictionary = json.load(fp)
                return dictionary
        else:
            print("File not found")
            return None

    def load_data_csv(self, filename):
        x = []
        y = []
        with open(filename, 'r') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for line in content:
                xi = line.split(',')[0]
                yi = line.split(',')[-1]
                x.append(xi)
                y.append(yi)
        logging.debug(x)
        logging.debug(y)
        return x, y

    def load_data_from_excel(self, filename):
        df = pd.read_excel(filename)
        lines = []
        tags = []
        for line in df[df.columns[1]]:
            lines.append(line)
        for tag in df[df.columns[3]]:
            tags.append(tag)

        return np.array(lines), np.array(tags)

