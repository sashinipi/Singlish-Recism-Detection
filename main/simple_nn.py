'''
Created on Apr 03, 2019

@author: dulan
'''

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from main.model import Model
from params import MISC, SNN, FILES
from main.logger import Logger


class SimpleNN(Model):
    def __init__(self):
        self.input_size = None
        super(SimpleNN, self).__init__()
        self.logger = Logger.get_logger(SNN.LOG_FILE_NAME)


    def training_stage(self):
        # messages = pd.read_csv(FILES.SEP_CSV_FILE_PATHS.format('all'), sep=',', names=["message", "label"])
        # self.classify.data_len = len(messages)
        # train_x, train_y, test_x, test_y = self.classify.split_data(messages['message'], messages['label'],
        #                                                             ratio=0.3)
        messages_train = pd.read_csv(FILES.SEP_CSV_FILE_PATHS.format('train'), sep=',', names=["message", "label"])
        messages_test = pd.read_csv(FILES.SEP_CSV_FILE_PATHS.format('test'), sep=',', names=["message", "label"])
        self.data_len = len(messages_train) + len(messages_test)
        train_x, train_y, test_x, test_y = messages_train['message'], messages_train['label'], messages_test['message'], \
                                           messages_test['label']

        features = self.train_feature_gen(train_x)
        self.input_size = features.shape[1]
        self.logger.info("Feature size:", self.input_size)
        self.pic_obj.save_obj(SNN.INPUT_FILENAME, self.input_size)
        self.create_model(self.input_size)

        label_train_float = np.array([self.trans_val(val) for val in train_y])
        self.train(features, label_train_float)

        test_features = self.get_features(test_x)
        label_test_float = np.array([self.trans_val(val) for val in test_y])
        self.evaluate(test_features, label_test_float)


        self.save_model(SNN.MODEL_FILENAME)
        self.save_transformers(SNN.TRANS_FILENAME)

    def load_values(self):
        self.create_model(self.pic_obj.load_obj(SNN.INPUT_FILENAME))
        self.load_model(SNN.MODEL_FILENAME)
        if self.bow_transformer is None or self.tfidf_transformer is None:
            self.load_transformers(SNN.TRANS_FILENAME)

    def predict_api(self, text):
        prediction = np.squeeze(self.predict(text))
        max_id = int(np.argmax(prediction))
        return MISC.CLASSES[max_id], prediction[max_id]

    def predict_cli(self):
        while(True):
            text = input("Input:")
            p_clas, conf = self.predict_api(text)
            self.logger.info("Predicted: {} Confidence: {}".format(p_clas, conf))

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
        self.model.fit(X, Y, epochs=20, batch_size=100)

    def predict(self, text):
        return self.model.predict(self.get_features([text]))

if __name__ == '__main__':
    snn = SimpleNN()
    is_train = False
    if is_train:
        snn.training_stage()
        snn.test_accuracy(snn.get_features)
    else:
        snn.load_values()
        snn.test_accuracy(snn.get_features)
        snn.predict_cli()