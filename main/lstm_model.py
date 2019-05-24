'''
Created on Mar 31, 2019

@author: dulan
'''

import numpy as np
import pandas as pd
import logging
import os.path as osp

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
# from params import GRUP as LSTMP

from main.model import Model
from params import FILES, MISC
from main.preprocess.singlish_preprocess import singlish_preprocess
from main.logger import Logger
from main.graph import Graph
from main.perfomance_test import PerformanceTest

class LSTMModel(Model):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.model_file = None
        self.model = None
        self.bow_transformer = None
        self.tfidf_transformer = None
        self.dictionary = None
        self.pre_pro = singlish_preprocess()
        self.logger = Logger.get_logger(LSTMP.LOG_FILE_NAME)
        self.graph_obj_1 = Graph(LSTMP.GRAPH_NAME)
        self.perf_test_o = PerformanceTest(LSTMP.PERF_TEST_NAME)

    def trandform_to_ngram(self, paragraph):
        if LSTMP.N_GRAM_LEN > 1:
            n_gram_para = []

            for e, word in enumerate(paragraph):
                n_gram_word = ''
                for i in range(LSTMP.N_GRAM_LEN):
                    if np.shape(paragraph)[0] > e + i:
                        n_gram_word += paragraph[e + i]
                    else:
                        n_gram_word = ''
                if n_gram_word is not '':
                    n_gram_para.append(n_gram_word + ' ')
            return  n_gram_para
        else:
            return paragraph

    def transform_to_dictionary_values(self, corpus_token: list, dictionary: dict) -> list:
        x_corpus = []
        for para in corpus_token:
            x_corpus.append(self.transform_to_dictionary_values_one(para)[0])

        return x_corpus

    def transform_to_dictionary_values_one(self, paragraph):
        x_corpus = []
        processed_para = self.pre_pro.pre_process(paragraph)
        n_gram_para = self.trandform_to_ngram(processed_para)
        x_corpus.append([self.dictionary[token] if token in self.dictionary else 1 for token in n_gram_para])
        return x_corpus

    def build_dictionary(self, corpus_token: list, dictionary_size=-1):
        word_frequency = {}
        dictionary = {}
        sentence = ""
        try:
            for sentence in corpus_token:
                list_of_words = sentence.split(' ')
                for e, word in enumerate(list_of_words):
                    if LSTMP.N_GRAM_LEN is not 1:
                        ngram = ''
                        for i in range(LSTMP.N_GRAM_LEN):
                            if len(list_of_words) > e + i:
                                ngram += list_of_words[e + i]
                            else:
                                ngram = ''
                        if ngram is not '':
                            if ngram in word_frequency:
                                word_frequency[ngram] += 1
                            else:
                                word_frequency[ngram] = 1
                    else:
                        if word in word_frequency:
                            word_frequency[word] += 1
                        else:
                            word_frequency[word] = 1
        except(AttributeError):
            print("SEN:{}".format(sentence))
            raise AttributeError
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

        self.logger.info("Model Architecture")
        self.logger.info(model.summary())
        return model

    def transform_class_to_one_hot_representation(self, classes: list):
        return np.array([LSTMP.DATA_SET_CLASSES[cls] for cls in classes])

    def train(self, x_train_corpus, y_train_corpus):
        pass

    def feature_gen(self, x_corpus):
        x_train_corpus = self.transform_to_dictionary_values(x_corpus, self.dictionary)
        return sequence.pad_sequences(x_train_corpus, maxlen=LSTMP.LSTM_MAX_WORD_COUNT)

    def train_n_test(self, x_train_corpus, y_train_corpus, x_test_corpus, y_test_corpus):
        self.dictionary = self.build_dictionary(x_train_corpus, dictionary_size=LSTMP.DICT_SIZE)
        self.pic_obj.save_obj(LSTMP.DICTIONARY_FILENAME, self.dictionary)

        x_train_corpus = self.feature_gen(x_train_corpus)
        y_train_corpus = self.transform_class_to_one_hot_representation(y_train_corpus)
        dictionary_length = len(self.dictionary) + 2

        self.log_n_print("Dictionary Length: {}".format(dictionary_length))

        # self.model = self.create_model(dictionary_length)

        # For Kfold split
        y_corpus_raw = ([0 if cls[0] == 1 else 1 for cls in y_train_corpus])

        k_fold = StratifiedKFold(n_splits=LSTMP.FOLDS_COUNT, shuffle=True, random_state=18)
        fold = 0
        best_overall_accuracy = 0
        for train_n_validation_indexes, test_indexes in k_fold.split(x_train_corpus, y_corpus_raw):
            self.model = self.create_model(dictionary_length)
            x_train_n_validation = x_train_corpus[train_n_validation_indexes]
            y_train_n_validation = y_train_corpus[train_n_validation_indexes]
            x_test = x_train_corpus[test_indexes]
            y_test = y_train_corpus[test_indexes]

            x_train, x_valid, y_train, y_valid = train_test_split(x_train_n_validation, y_train_n_validation,
                                                                  test_size=LSTMP.VALIDATION_TEST_SIZE)#, random_state=94

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
                self.log_n_print("Epoch: {}/{} | Fold {}/{}".format(epoch+1, LSTMP.MAX_EPOCHS, fold, LSTMP.FOLDS_COUNT))
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

                self.logger.info("Fold: {} Epoch: {} | loss: {} - acc: {} - val_loss: {} - val_acc: {}".format(
                    fold, epoch,
                    history.history['loss'][0],
                    history.history['acc'][0],
                    history.history['val_loss'][0],
                    history.history['val_acc'][0]
                ))

                # select best epoch and save to disk
                if accuracy >= best_accuracy and loss < best_loss + 0.01:
                    self.log_n_print("Saving model....")
                    self.model.save("%s/model_fold_%d.h5" % (LSTMP.OUTPUT_DIR, fold))
                    best_accuracy = accuracy
                    best_loss = loss
                    best_epoch = epoch

                    evaluation = self.model.evaluate(x=x_test, y=y_test)
                    self.log_n_print(
                        "========== Fold {} : Accuracy for split test data =========".format(fold))
                    self.log_n_print("Accuracy: %f" % evaluation[1])


            del self.model
            self.model = load_model("%s/model_fold_%d.h5" % (LSTMP.OUTPUT_DIR, fold))
            self.log_n_print(
                "========== Fold {} : Accuracy for test data set in data/output_test.csv =========".format(fold))
            total_acc = self.test_accuracy(self.feature_gen)
            if best_overall_accuracy < total_acc:
                best_overall_accuracy = total_acc
                self.log_n_print("Model saved; Best accuracy: {}".format(best_overall_accuracy))
                self.model.save("%s/%s" % (LSTMP.OUTPUT_DIR, LSTMP.MODEL_FILENAME))

            # self.test_accuracy_lstm(x_test_corpus, y_test_corpus)
            self.graph_obj_1.set_lables('Fold-{}'.format(fold+1), 'No of Epoches', 'error', 'Percentage')
            self.graph_obj_1.set_legends('Loss', 'Training-Acc', 'Validation-Acc')
            self.graph_obj_1.plot_3sub(epoch_history['loss'], epoch_history['acc'],
                                       epoch_history['val_acc'], 'lstm-graph-fold-{}'.format(fold))

            fold += 1
            if fold == LSTMP.BREAK_AFTER_FOLD:
                break

            del self.model

    def predict(self, text):
        x_corpus = self.transform_to_dictionary_values_one(text)
        x_corpus_padded = sequence.pad_sequences(x_corpus, maxlen=LSTMP.LSTM_MAX_WORD_COUNT)
        return np.squeeze(self.model.predict([x_corpus_padded]))

    def train_now(self):
        messages_train = self.load_csv(FILES.SEP_CSV_FILE_PATHS.format('train'))
        messages_test = self.load_csv(FILES.SEP_CSV_FILE_PATHS.format('test'))
        train_x, train_y, test_x, test_y = messages_train["message"], messages_train["label"], messages_test["message"], messages_test["label"]

        self.train_n_test(train_x, train_y, test_x, test_y)

        self.model = load_model("%s/%s" % (LSTMP.OUTPUT_DIR, LSTMP.MODEL_FILENAME))
        acc = self.test_accuracy(self.feature_gen)
        self.model.save(osp.join(LSTMP.OUTPUT_DIR, LSTMP.MODEL_FILENAME_ACC.format(acc)))


    def load_values(self):
        self.dictionary = self.pic_obj.load_obj(LSTMP.DICTIONARY_FILENAME)
        dictionary_length = len(self.dictionary) + 2
        print(dictionary_length)
        self.model = self.create_model(dictionary_length)
        self.load_model(LSTMP.MODEL_FILENAME)
        acc = self.test_accuracy(self.feature_gen)
        print("Loaded Model Accuracy: {}".format(acc))
        # self.perf_test_o.perform_test(self.predict)

    def predict_cli(self):
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
    train = True
    if train:
        lstm_obj.train_now()
        lstm_obj.predict_cli()
    else:
        lstm_obj.load_values()
        lstm_obj.predict_cli()
