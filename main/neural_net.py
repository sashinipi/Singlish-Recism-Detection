'''
Created on Apr 19, 2019

@author: dulan
'''
'''
Created on Apr 03, 2019

@author: dulan
'''

from params import FILES

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from main.model import Model
from params import DIR, MISC, NN
import logging
import os.path as osp

from keras.models import load_model

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

class NeuralNet(Model):
    trans_file_name = 'trans2'
    model_file_name = 'model2.h5'
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.input_size = None
        logging.basicConfig(filename=osp.join(DIR.LOGS_DIR, 'nn.log'), level=logging.INFO)

    def training_stage(self):
        messages_train = pd.read_csv(FILES.SEP_CSV_FILE_PATHS.format('all'), sep=',', names=["message", "label"])
        train_x, train_y= messages_train["message"], messages_train["label"]

        f_train_x = self.train_feature_gen(train_x)
        f_train_y = np.array([self.trans_val(val) for val in train_y])

        self.input_size = f_train_x.shape[1]
        print("Feature size:", self.input_size)
        self.pic_obj.save_obj('size-nn', self.input_size)
        self.create_model(self.input_size)

        self.train(f_train_x, f_train_y)

        # messages = pd.read_csv(FILES.CSV_FILE_PATH, sep=',', names=["message", "label"])
        # self.classify.data_len = len(messages)
        # train_x, train_y, test_x, test_y = self.classify.split_data(messages['message'], messages['label'],
        #                                                             ratio=0.3)
        # features = self.train_feature_gen(train_x)
        # self.input_size = features.shape[1]
        # print("Feature size:", self.input_size)
        # self.pic_obj.save_obj('size', self.input_size)
        # self.create_model(self.input_size)
        #
        # label_train_float = np.array([self.trans_val(val) for val in train_y])
        # print(len(label_train_float))
        # print(label_train_float[5:15])
        # self.train(features, label_train_float)
        #
        # test_features = self.get_features(test_x)
        # label_test_float = np.array([self.trans_val(val) for val in test_y])
        # self.evaluate(test_features, label_test_float)
        # self.save_model(NeuralNet.model_file_name)
        # self.save_transformers(NeuralNet.trans_file_name)

    def load_values(self):
        self.create_model(self.pic_obj.load_obj('size-nn'))
        self.load_model(NeuralNet.model_file_name)
        if self.bow_transformer is None or self.tfidf_transformer is None:
            self.load_transformers(NeuralNet.trans_file_name)

    def predict_api(self, text):
        prediction = np.squeeze(self.predict(text))
        max_id = int(np.argmax(prediction))
        return MISC.CLASSES[max_id], prediction[max_id]

    def predict_cli(self):
        while(True):
            text = input("Input:")
            # prediction = np.squeeze(self.predict(text))
            # max_id = int(np.argmax(prediction))
            # print(prediction)
            p_clas, conf = self.predict_api(text)
            print("Predicted: {} Confidence: {}".format(p_clas, conf))

    def create_model(self, input_dim):
        # create model
        model = Sequential()
        model.add(Dense(256, input_dim=input_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def train(self, x_corpus, y_corpus):
        print("DIM X: {}\nDIM Y:{}".format(x_corpus.shape, y_corpus.shape))
        k_fold = StratifiedKFold(n_splits=NN.FOLDS_COUNT, shuffle=True, random_state=18)
        y_corpus_raw = ([0 if cls[0] == 1 else 1 for cls in y_corpus])
        fold = 0
        for train_n_validation_indexes, test_indexes in k_fold.split(x_corpus, y_corpus_raw):
            x_train_n_validation = x_corpus[train_n_validation_indexes]
            y_train_n_validation = y_corpus[train_n_validation_indexes]
            x_test = x_corpus[test_indexes]
            y_test = y_corpus[test_indexes]

            # print(x_test)

            x_train, x_valid, y_train, y_valid = train_test_split(x_train_n_validation, y_train_n_validation,
                                                                  test_size=NN.VALIDATION_TEST_SIZE, random_state=94)

            best_accuracy = 0
            best_loss = 100000
            best_epoch = 0

            epoch_history = {
                'acc': [],
                'val_acc': [],
                'loss': [],
                'val_loss': [],
            }

            # for each epoch
            for epoch in range(NN.MAX_EPOCHS):
                print("Epoch: {}/{} | Fold {}/{}".format(epoch + 1, NN.MAX_EPOCHS, fold, NN.FOLDS_COUNT))
                logging.info("Fold: %d/%d" % (fold, NN.FOLDS_COUNT))
                logging.info("Epoch: %d/%d" % (epoch, NN.MAX_EPOCHS))
                history = self.model.fit(x=x_train, y=y_train, epochs=1, batch_size=1,
                                         validation_data=(x_valid, y_valid),
                                         verbose=1, shuffle=False)

                # get validation (test) accuracy and loss
                accuracy = history.history['val_acc'][0]
                loss = history.history['val_loss'][0]

                # set epochs' history
                epoch_history['acc'].append(history.history['acc'][0])
                epoch_history['val_acc'].append(history.history['val_acc'][0])
                epoch_history['loss'].append(history.history['loss'][0])
                epoch_history['val_loss'].append(history.history['val_loss'][0])

                # select best epoch and save to disk
                if accuracy >= best_accuracy and loss < best_loss + 0.01:
                    logging.info("Saving model")
                    self.model.save("%s/model_fold_%d.h5" % (NN.OUTPUT_DIR, fold))

                    best_accuracy = accuracy
                    best_loss = loss
                    best_epoch = epoch
                # end of epoch

            # del self.model
            model = load_model("%s/model_fold_%d.h5" % (NN.OUTPUT_DIR, fold))

            evaluation = model.evaluate(x=x_test, y=y_test)
            logging.info("Accuracy: %f" % evaluation[1])
            fold += 1


        # Fit the model
        # self.model.fit(X, Y, epochs=20, batch_size=100)

    def predict(self, text):
        return self.model.predict(self.get_features([text]))

if __name__ == '__main__':
    snn = NeuralNet()
    is_train = True
    if is_train:
        snn.training_stage()
    else:
        snn.load_values()
        snn.predict_cli()