'''
Created on Apr 01, 2019

@author: dulan
'''

import pandas as pd
# import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer

from params import FILES
from sklearn.metrics import classification_report
from random import shuffle
from main.preprocess.singlish_preprocess import singlish_preprocess
from main.pickel_helper import PickelHelper

class classify(object):
    def __init__(self):
        self.singlish_preprocess_obj = singlish_preprocess()
        self.data_len = None
        self.pick_obj = PickelHelper()
        self.model = None
        self.bow_transformer = None
        self.tfidf_transformer = None

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

    def test(self, test_x, test_y):
        messages_bow = self.bow_transformer.transform(test_x)
        messages_tfidf = self.tfidf_transformer.transform(messages_bow)
        predictions = self.model.predict(messages_tfidf)
        print(classification_report(predictions, test_y))

    def train(self, train_x, train_y):
        raise NotImplementedError

    def train_test(self):
        messages = pd.read_csv(FILES.CSV_FILE_PATH, sep=',', names=["message", "label"])
        self.data_len = len(messages)
        train_x, train_y, test_x, test_y = self.split_data(messages['message'], messages['label'], ratio=0.3)
        self.train(train_x, train_y)
        self.test(test_x, test_y)

    def save_models(self, names):
        self.pick_obj.save_obj(names.MODEL_FILENAME, self.model)
        self.pick_obj.save_obj(names.BOW_FILENAME, self.bow_transformer)
        self.pick_obj.save_obj(names.TFIDF_FILENAME, self.tfidf_transformer)
        self.pick_obj.save_obj(names.INPUT_FILENAME, self.data_len)

    def load_models(self, names):
        self.model = self.pick_obj.load_obj(names.MODEL_FILENAME)
        self.bow_transformer = self.pick_obj.load_obj(names.BOW_FILENAME)
        self.tfidf_transformer = self.pick_obj.load_obj(names.TFIDF_FILENAME)
        self.data_len = self.pick_obj.load_obj(names.INPUT_FILENAME)
