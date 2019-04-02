'''
Created on Apr 01, 2019

@author: dulan
'''

import pandas as pd
# import seaborn as sns
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from params import FILES
from sklearn.metrics import classification_report
from random import shuffle
import random
from main.singlish_preprocess import singlish_preprocess

class classify(object):
    def __init__(self):
        self.singlish_preprocess_obj = singlish_preprocess()
        self.data_len = None

    def text_process(self, mess):
        return self.singlish_preprocess_obj.pre_process(mess)

    def split_data(self, x, y, ratio=0.2):

        test_x = []
        test_y = []
        train_x = []
        train_y = []

        count_r = 0
        count_n = 0
        test_size = int(self.data_len * ratio)
        ids = list(range(self.data_len))
        shuffle(ids)
        for i in ids:
            if y[i] == 'Racist' and count_r < test_size/2:
                test_x.append(x[i])
                test_y.append(y[i])
                count_r += 1
                continue

            if y[i] == 'Neutral' and count_n < test_size/2:
                test_x.append(x[i])
                test_y.append(y[i])
                count_n += 1
                continue

            train_x.append(x[i])
            train_y.append(y[i])

        return train_x, train_y, test_x, test_y

    def test(self, bow_transformer, test_x, test_y, model):
        messages_bow = bow_transformer.transform(test_x)

        tfidf_transformer = TfidfTransformer().fit(messages_bow)

        messages_tfidf = tfidf_transformer.transform(messages_bow)
        predictions = model.predict(messages_tfidf)
        print(classification_report(predictions, test_y))

    def train(self, train_x, train_y):
        raise NotImplementedError

    def main(self):
        messages = pd.read_csv(FILES.CSV_FILE_PATH, sep=',', names=["message", "label"])
        print(len(messages))
        self.data_len = len(messages)

        train_x, train_y, test_x, test_y = self.split_data(messages['message'], messages['label'], ratio=0.3)

        model , bow_transformer= self.train(train_x, train_y)
        self.test(bow_transformer, test_x, test_y, model)
