'''
Created on Mar 31, 2019

@author: dulan
'''
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import regularizers
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from params import LSTMP
from keras.preprocessing import sequence

from main.model import Model
import logging
from params import FILES, MISC, DIR
from main.preprocess.singlish_preprocess import singlish_preprocess
import os.path as osp
from main.logger import Logger

class LSTMModel(Model):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.model_file = None
        self.model = None
        self.bow_transformer = None
        self.tfidf_transformer = None
        self.dictionary = None
        self.pre_pro = singlish_preprocess()
        # print("LOGS DIR:{}".format(DIR.LOGS_DIR))
        # logging.basicConfig(filename=osp.join(DIR.LOGS_DIR, 'lstm.log'),
        #                     filemode='w', level=logging.INFO)

        self.logger = Logger.get_logger(LSTMP.LOG_FILE_NAME)


    def transform_to_dictionary_values(self, corpus_token: list, dictionary: dict) -> list:
        x_corpus = []
        for para in corpus_token:
            processed_para = self.pre_pro.pre_process(para)
            x_corpus.append([dictionary[token] if token in self.dictionary else 1 for token in processed_para])

        return x_corpus

    def transform_to_dictionary_values_one(self, paragraph):
        x_corpus = []
        processed_para = self.pre_pro.pre_process(paragraph)
        x_corpus.append([self.dictionary[token] if token in self.dictionary else 1 for token in processed_para])
        return x_corpus

    def build_dictionary(self, corpus_token: list, dictionary_size=-1):
        word_frequency = {}
        dictionary = {}

        for sentence in corpus_token:
            print(sentence)
            for word in sentence.split(' '):
                if word in word_frequency:
                    word_frequency[word] += 1
                else:
                    word_frequency[word] = 1

        frequencies = list(word_frequency.values())
        unique_words = list(word_frequency.keys())

        # sort words by its frequency
        frequency_indexes = np.argsort(frequencies)[::-1]  # reverse for descending
        for index, frequency_index in enumerate(frequency_indexes):
            # 0 is not used and 1 is for UNKNOWN
            if index == dictionary_size:
                break
            dictionary[unique_words[frequency_index]] = index + 2

        return dictionary

    def create_model(self, input_dim):
        # ################## Deep Neural Network Model ###################### #
        model = Sequential()
        model.add(Embedding(input_dim=input_dim, output_dim=60, input_length=LSTMP.LSTM_MAX_WORD_COUNT))
        model.add(LSTM(units=600))
        model.add(Dense(units=LSTMP.LSTM_MAX_WORD_COUNT, activation='tanh', kernel_regularizer=regularizers.l2(0.04),
                        activity_regularizer=regularizers.l2(0.015)))
        model.add(Dense(units=LSTMP.LSTM_MAX_WORD_COUNT, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                        bias_regularizer=regularizers.l2(0.01)))
        model.add(Dense(units=2, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))
        adam_optimizer = Adam(lr=0.001, decay=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

        print(model.summary())
        return model


    def transform_class_to_one_hot_representation(self, classes: list):
        return np.array([LSTMP.DATA_SET_CLASSES[cls] for cls in classes])

    def train(self, x_train_corpus, y_train_corpus):
        pass

    def train_n_test(self, x_train_corpus, y_train_corpus, x_test_corpus, y_test_corpus):
        self.dictionary = self.build_dictionary(x_train_corpus, dictionary_size=LSTMP.DICT_SIZE)
        self.pic_obj.save_obj(LSTMP.DICTIONARY_FILENAME, self.dictionary)
        x_train_corpus = self.transform_to_dictionary_values(x_train_corpus, self.dictionary)
        x_train_corpus = sequence.pad_sequences(x_train_corpus, maxlen=LSTMP.LSTM_MAX_WORD_COUNT)

        y_train_corpus = self.transform_class_to_one_hot_representation(y_train_corpus)
        dictionary_length = len(self.dictionary) + 2
        print("Dictionary Length: {}".format(dictionary_length))
        self.model = self.create_model(dictionary_length)
        y_corpus_raw = ([0 if cls[0] == 1 else 1 for cls in y_train_corpus])

        k_fold = StratifiedKFold(n_splits=LSTMP.FOLDS_COUNT, shuffle=True, random_state=18)
        fold = 0
        for train_n_validation_indexes, test_indexes in k_fold.split(x_train_corpus, y_corpus_raw):
            x_train_n_validation = x_train_corpus[train_n_validation_indexes]
            y_train_n_validation = y_train_corpus[train_n_validation_indexes]
            x_test = x_train_corpus[test_indexes]
            y_test = y_train_corpus[test_indexes]

            x_train, x_valid, y_train, y_valid = train_test_split(x_train_n_validation, y_train_n_validation,
                                                                  test_size=LSTMP.VALIDATION_TEST_SIZE, random_state=94)

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
            for epoch in range(LSTMP.MAX_EPOCHS):
                print("Epoch: {}/{} | Fold {}/{}".format(epoch+1, LSTMP.MAX_EPOCHS, fold, LSTMP.FOLDS_COUNT))
                logging.info("Fold: %d/%d" % (fold, LSTMP.FOLDS_COUNT))
                logging.info("Epoch: %d/%d" % (epoch, LSTMP.MAX_EPOCHS))
                history = self.model.fit(x=x_train, y=y_train, epochs=1, batch_size=LSTMP.BATCH_SIZE, validation_data=(x_valid, y_valid),
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
                    print("Saving model")
                    print("fold_{}\nEpoch_{}\nAcc_{}\nLoss_{}\n".format(fold, epoch, accuracy, loss))
                    self.model.save(osp.join(LSTMP.OUTPUT_DIR, LSTMP.MODEL_FILENAME))
                    best_accuracy = accuracy
                    best_loss = loss
                    best_epoch = epoch

                    evaluation = self.model.evaluate(x=x_test, y=y_test)
                    print("Accuracy: %f" % evaluation[1])
                    logging.info("Accuracy: %f" % evaluation[1])

            self.model = load_model(osp.join(LSTMP.OUTPUT_DIR, LSTMP.MODEL_FILENAME))
            self.test_accuracy(x_test_corpus, y_test_corpus)
            fold += 1


    def test_accuracy(self, x_test_corpus, y_test_corpus):
        print('Final Accuracy')
        x_corpus = sequence.pad_sequences(self.transform_to_dictionary_values(x_test_corpus, self.dictionary),
                                          maxlen=LSTMP.LSTM_MAX_WORD_COUNT)
        y_corpus = self.transform_class_to_one_hot_representation(y_test_corpus)
        return self.evaluate(x_corpus, y_corpus)

    def main(self):
        train = True
        if train:
            messages_train = pd.read_csv(FILES.SEP_CSV_FILE_PATHS.format('train'), sep=',', names=["message", "label"])
            messages_test = pd.read_csv(FILES.SEP_CSV_FILE_PATHS.format('test'), sep=',', names=["message", "label"])
            train_x, train_y, test_x, test_y = messages_train["message"], messages_train["label"], messages_test["message"], messages_test["label"]

            self.train_n_test(train_x, train_y, test_x, test_y)

            # self.dictionary = self.pic_obj.load_obj(LSTMP.DICTIONARY_FILENAME)
            # dictionary_length = len(self.dictionary) + 2
            # self.model = self.create_model(dictionary_length)
            # self.load_model(LSTMP.MODEL_FILENAME)

            acc = self.test_accuracy(test_x, test_y)
            self.model.save(osp.join(LSTMP.OUTPUT_DIR, LSTMP.MODEL_FILENAME_ACC.format(acc)))

        else:
            self.dictionary = self.pic_obj.load_obj(LSTMP.DICTIONARY_FILENAME)
            dictionary_length = len(self.dictionary) + 2
            print(dictionary_length)
            self.model = self.create_model(dictionary_length)
            self.load_model(LSTMP.MODEL_FILENAME)
            while(True):
                text = input("Input:")
                x_corpus = self.transform_to_dictionary_values_one(text)
                x_corpus = sequence.pad_sequences(x_corpus, maxlen=LSTMP.LSTM_MAX_WORD_COUNT)
                prediction = np.squeeze(self.model.predict([x_corpus]))
                max_id = int(np.argmax(prediction))
                print(prediction)
                print("Predicted: {} Confidence: {}".format(MISC.CLASSES[max_id], prediction[max_id]))


if __name__ == '__main__':

    lstm_obj = LSTMModel()
    lstm_obj.main()
