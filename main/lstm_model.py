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
from params import LSTM

from main.model import model
import logging

class lstm_model(model):
    def __init__(self):
        super(lstm_model, self).__init__()
        self.model_file = None
        self.model = self.create_model()

    def create_model(self):
        # ################## Deep Neural Network Model ###################### #
        model = Sequential()
        model.add(Embedding(input_dim=dictionary_length, output_dim=60, input_length=LSTM.LSTM_MAX_WORD_COUNT))
        model.add(LSTM(units=600))
        model.add(Dense(units=LSTM.LSTM_MAX_WORD_COUNT, activation='tanh', kernel_regularizer=regularizers.l2(0.04),
                        activity_regularizer=regularizers.l2(0.015)))
        model.add(Dense(units=LSTM.LSTM_MAX_WORD_COUNT, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                        bias_regularizer=regularizers.l2(0.01)))
        model.add(Dense(units=2, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))
        adam_optimizer = Adam(lr=0.001, decay=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

        print(model.summary())
        return model
        # ################## Deep Neural Network Model ###################### #

    def train(self, x_corpus, y_corpus):
        # splitting data for 5-fold cross validation
        k_fold = StratifiedKFold(n_splits=LSTM.FOLDS_COUNT, shuffle=True, random_state=18)
        fold = 0
        for train_n_validation_indexes, test_indexes in k_fold.split(x_corpus, y_corpus_raw):
            x_train_n_validation = x_corpus[train_n_validation_indexes]
            y_train_n_validation = y_corpus[train_n_validation_indexes]
            x_test = x_corpus[test_indexes]
            y_test = y_corpus[test_indexes]

            # train and validation data sets
            x_train, x_valid, y_train, y_valid = train_test_split(x_train_n_validation, y_train_n_validation,
                                                                  test_size=LSTM.VALIDATION_TEST_SIZE, random_state=94)

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
            for epoch in range(LSTM.MAX_EPOCHS):
                logging.info("Fold: %d/%d" % (fold, LSTM.FOLDS_COUNT))
                logging.info("Epoch: %d/%d" % (epoch, LSTM.MAX_EPOCHS))
                history = self.model.fit(x=x_train, y=y_train, epochs=1, batch_size=1, validation_data=(x_valid, y_valid),
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
                    self.model.save("%s/model_fold_%d.h5" % (LSTM.OUTPUT_DIR, fold))

                    best_accuracy = accuracy
                    best_loss = loss
                    best_epoch = epoch
                # end of epoch