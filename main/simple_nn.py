'''
Created on Apr 03, 2019

@author: dulan
'''

from params import FILES

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import categorical_accuracy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from params import DIR
import os.path as osp
from main.model import Model
from params import DIR, MISC

class SimpleNN(Model):
    # classes = ['Racist', 'Neutral']
    trans_file_name = 'trans'
    model_file_name = 'model.h5'
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.input_size = None

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
            self.save_model(SimpleNN.model_file_name)
            self.save_transformers(SimpleNN.trans_file_name)
        else:
            self.create_model(self.pic_obj.load_obj('size'))
            self.load_model(SimpleNN.model_file_name)
            if self.bow_transformer is None or self.tfidf_transformer is None:
                self.load_transformers(SimpleNN.trans_file_name)
            while(True):
                text = input("Input:")
                prediction = np.squeeze(self.model.predict(self.get_features([text])))
                max_id = int(np.argmax(prediction))
                print(prediction)
                print("Predicted: {} Confidence: {}".format(MISC.CLASSES[max_id], prediction[max_id]))



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





if __name__ == '__main__':
    snn = SimpleNN()
    snn.main()