'''
Created on Apr 03, 2019

@author: dulan
'''
from main.classify import classify
from params import FILES

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import categorical_accuracy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from main.pickel_helper import PickelHelper
from params import DIR
import os.path as osp


class SimpleNN(object):
    classes = ['Racist', 'Neutral']
    trans_file_name = 'trans'
    model_file_name = 'model.h5'
    def __init__(self):
        self.classify = classify()
        self.model = None  # self.create_model(input_dim)
        self.bow_transformer = None
        self.tfidf_transformer = None
        self.input_size = None
        self.pic_obj = PickelHelper()

    def main(self):
        train = False
        if train:
            messages = pd.read_csv(FILES.CSV_FILE_PATH, sep=',', names=["message", "label"])
            self.classify.data_len = len(messages)
            train_x, train_y, test_x, test_y = self.classify.split_data(messages['message'], messages['label'],
                                                                        ratio=0.3)
            features = self.train_feature_gen(train_x)
            self.input_size = features.shape[1]
            print("Feature size:", self.input_size)
            self.pic_obj.save_obj('size', self.input_size)
            self.create_model(self.input_size)

            label_train_float = np.array([self.trans_val(val) for val in train_y])
            print(len(label_train_float))
            print(label_train_float[5:15])
            self.train(features, label_train_float)

            test_features = self.get_features(test_x)
            label_test_float = np.array([self.trans_val(val) for val in test_y])
            self.evaluate(test_features, label_test_float)
            self.save_model()
            self.save_transformers()
        else:
            self.create_model(self.pic_obj.load_obj('size'))
            self.load_model()
            while(True):
                text = input("Input:")
                prediction = np.squeeze(self.model.predict(self.get_features([text])))
                max_id = int(np.argmax(prediction))
                print(prediction)
                print("Predicted: {} Confidence: {}".format(SimpleNN.classes[max_id], prediction[max_id]))


    def get_features(self, data):
        if self.bow_transformer is None or self.tfidf_transformer is None:
            self.load_transformers()
        bw_msg = self.bow_transformer.transform(data)
        tfidef_msg = self.tfidf_transformer.transform(bw_msg)
        return tfidef_msg

    def evaluate(self, X, Y):
        # evaluate the model
        scores = self.model.evaluate(X, Y)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    def trans_val(self, val):
        if val == SimpleNN.classes[0]:
            return [1, 0]
        else:
            return [0, 1]

    def create_model(self, input_dim):
        # create model
        model = Sequential()
        model.add(Dense(256, input_dim=input_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def train(self, X, Y ):
        # Fit the model
        self.model.fit(X, Y, epochs=20, batch_size=100)

    def train_feature_gen(self, msg_train):
        bow_transformer = CountVectorizer(analyzer=self.classify.text_process).fit(msg_train)
        messages_bow = bow_transformer.transform(msg_train)
        tfidf_transformer = TfidfTransformer().fit(messages_bow)
        messages_tfidf = tfidf_transformer.transform(messages_bow)

        self.bow_transformer = bow_transformer
        self.tfidf_transformer = tfidf_transformer
        return messages_tfidf

    def save_model(self, model_name=model_file_name):
        self.model.save_weights(osp.join(DIR.DEF_SAV_LOC, model_name))

    def load_model(self, weights_file=model_file_name):
        self.model.load_weights(osp.join(DIR.DEF_SAV_LOC, weights_file))

    def save_transformers(self):
        transform = {'bow': self.bow_transformer, 'tfidf': self.tfidf_transformer}
        self.pic_obj.save_obj(SimpleNN.trans_file_name, transform)

    def load_transformers(self):
        transform = self.pic_obj.load_obj(SimpleNN.trans_file_name)
        self.tfidf_transformer = transform['tfidf']
        self.bow_transformer = transform['bow']

if __name__ == '__main__':
    snn = SimpleNN()
    snn.main()