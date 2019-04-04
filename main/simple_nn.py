'''
Created on Apr 03, 2019

@author: dulan
'''
from main.classify import classify
from params import FILES

from nltk.corpus import stopwords
import pandas as pd
import string
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import categorical_accuracy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import time

class SimpleNN(object):
    classes = ['Racist', 'Neutral']
    def __init__(self):
        self.classify = classify()
        self.model = None  # self.create_model(input_dim)
        self.bow_transformer = None
        self.tfidf_transformer = None

    def main(self):
        train = True
        messages = pd.read_csv(FILES.CSV_FILE_PATH, sep=',', names=["message", "label"])
        self.classify.data_len = len(messages)
        train_x, train_y, test_x, test_y = self.classify.split_data(messages['message'], messages['label'], ratio=0.3)

        features = self.train_feature_gen(train_x)
        size = features.shape[1]
        print("Feature size:", size)
        self.create_model(size)
        # if train:
        label_train_float = np.array([self.trans_val(val) for val in train_y])
        print(len(label_train_float))
        print(label_train_float[5:15])
        self.train(features, label_train_float)

        test_features = self.get_features(test_x)
        label_test_float = np.array([self.trans_val(val) for val in test_y])
        self.evaluate(test_features, label_test_float)
        self.save_model()
    # else:

        self.load_model()
        class_values = ["Racist", "Neutral"]
        while(True):

            text = input("Input:")
            prediction = np.squeeze(self.model.predict(self.get_features([text])))
            max_id = int(np.argmax(prediction))
            print(prediction)
            print("Predicted: {} Confidence: {}".format(class_values[max_id], prediction[max_id]))

        # snn = simple_nn()
        # snn.train(msg_train, label_train)

    def save_model(self, model_name='model'):
        # self.model.save(model_name)
        self.model.save_weights(model_name+"_{}.h5".format(time.time()))

    def get_features(self, data):
        bw_msg = self.bow_transformer.transform(data)
        tfidef_msg = self.tfidf_transformer.transform(bw_msg)
        return tfidef_msg

    def evaluate(self, X, Y):
        # evaluate the model
        scores = self.model.evaluate(X, Y)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    def trans_val(self, val):
        if val == 'Racist':
            return [1, 0]
        else:
            return [0, 1]

    def load_model(self, weights_file="model.h5"):
       self.model.load_weights(weights_file)

    def create_model(self, input_dim):
        # create model
        model = Sequential()
        model.add(Dense(256, input_dim=input_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))

        # Compile model
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def train(self, X, Y ):
        # Fit the model
        self.model.fit(X, Y, epochs=20, batch_size=100)

    # def test_feature_gen(self, msg_test):
    #     messages_bow = self.bow_transformer.transform(msg_test)
    #     tfidf_transformer = TfidfTransformer().fit(messages_bow)
    #     return tfidf_transformer.transform(messages_bow)

    def train_feature_gen(self, msg_train):
        bow_transformer = CountVectorizer(analyzer=self.classify.text_process).fit(msg_train)
        messages_bow = bow_transformer.transform(msg_train)
        tfidf_transformer = TfidfTransformer().fit(messages_bow)
        messages_tfidf = tfidf_transformer.transform(messages_bow)

        self.bow_transformer = bow_transformer
        self.tfidf_transformer = tfidf_transformer
        return messages_tfidf

if __name__ == '__main__':
    snn = SimpleNN()
    snn.main()