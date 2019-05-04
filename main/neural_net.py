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

from keras.models import load_model

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from main.logger import Logger
from main.graph import Graph
from main.perfomance_test import PerformanceTest

class NeuralNet(Model):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.input_size = None
        self.logger = Logger.get_logger(NN.LOG_FILE_NAME)
        self.graph_obj_1 = Graph('lstm-graph')
        self.perf_test_o = PerformanceTest('Neural-Net')

    def training_stage(self):
        messages_train = pd.read_csv(FILES.SEP_CSV_FILE_PATHS.format('train'), sep=',', names=["message", "label"])
        train_x, train_y= messages_train["message"], messages_train["label"]

        f_train_x = self.train_feature_gen(train_x)
        f_train_y = np.array([self.trans_val(val) for val in train_y])

        self.input_size = f_train_x.shape[1]
        self.logger.info("Feature size:", self.input_size)
        self.pic_obj.save_obj(NN.INPUT_FILENAME, self.input_size)
        self.model = self.create_model(self.input_size)

        self.save_transformers(NN.TRANS_FILENAME)

        self.train(f_train_x, f_train_y)

    def load_values(self):
        self.model = self.create_model(self.pic_obj.load_obj(NN.INPUT_FILENAME))
        self.load_model(NN.MODEL_FILENAME)
        if self.bow_transformer is None or self.tfidf_transformer is None:
            self.load_transformers(NN.TRANS_FILENAME)

    def predict_api(self, text):
        prediction = np.squeeze(self.predict(text))
        max_id = int(np.argmax(prediction))
        return MISC.CLASSES[max_id], prediction[max_id]

    def predict_cli(self):
        while(True):
            text = input("Input:")
            p_class, conf = self.predict_api(text)
            self.logger.info("Predicted: {} Confidence: {}".format(p_class, conf))

    def create_model(self, input_dim):
        # create model
        model = Sequential()
        model.add(Dense(256, input_dim=input_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.logger.info("Model Architecture")
        self.logger.info(model.summary())
        return model

    def train(self, x_corpus, y_corpus):
        print("DIM X: {}\nDIM Y:{}".format(x_corpus.shape, y_corpus.shape))
        k_fold = StratifiedKFold(n_splits=NN.FOLDS_COUNT, shuffle=True, random_state=18)
        y_corpus_raw = ([0 if cls[0] == 1 else 1 for cls in y_corpus])
        fold = 0
        best_overall_accuracy = 0
        for train_n_validation_indexes, test_indexes in k_fold.split(x_corpus, y_corpus_raw):
            x_train_n_validation = x_corpus[train_n_validation_indexes]
            y_train_n_validation = y_corpus[train_n_validation_indexes]
            x_test = x_corpus[test_indexes]
            y_test = y_corpus[test_indexes]

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
                self.logger.info("Epoch: {}/{} | Fold {}/{}".format(epoch + 1, NN.MAX_EPOCHS, fold, NN.FOLDS_COUNT))

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

                self.logger.info("Fold: {} Epoch: {} | loss: {} - acc: {} - val_loss: {} - val_acc: {}".format(
                    fold, epoch,
                    history.history['loss'][0],
                    history.history['acc'][0],
                    history.history['val_loss'][0],
                    history.history['val_acc'][0]
                ))

                # select best epoch and save to disk
                if accuracy >= best_accuracy and loss < best_loss + 0.01:
                    self.logger.info("Saving model....")
                    self.model.save("%s/model_fold_%d.h5" % (NN.OUTPUT_DIR, fold))
                    best_accuracy = accuracy
                    best_loss = loss
                    best_epoch = epoch

                    logging.info(
                        "========== Fold {} : Accuracy for split test data =========".format(fold))

                    evaluation = self.model.evaluate(x=x_test, y=y_test)
                    logging.info("Accuracy: %f" % evaluation[1])

            del self.model
            self.model = load_model("%s/model_fold_%d.h5" % (NN.OUTPUT_DIR, fold))
            logging.info("========== Fold {} : Accuracy for test data set in data/output_test.csv =========".format(fold))
            total_acc = self.test_accuracy(self.get_features)
            if best_overall_accuracy < total_acc:
                best_overall_accuracy = total_acc
                self.logger.info("Model saved; Best accuracy: {}".format(best_overall_accuracy))
                self.model.save("%s/%s" % (NN.OUTPUT_DIR, NN.MODEL_FILENAME))

            # self.test_accuracy_lstm(x_test_corpus, y_test_corpus)
            self.graph_obj_1.set_lables('Neural Net Fold-{}'.format(fold + 1), 'No of Epoches', 'error', 'Percentage')
            self.graph_obj_1.set_legends('Loss', 'Training-Acc', 'Validation-Acc')
            self.graph_obj_1.plot_3sub(epoch_history['loss'], epoch_history['acc'],
                                       epoch_history['val_acc'], 'nn-graph-fold-{}'.format(fold))

            del self.model
            self.model = self.create_model(self.input_size)

            fold += 1

    def predict(self, text):
        return self.model.predict(self.get_features([text]))

if __name__ == '__main__':
    snn = NeuralNet()
    is_train = False
    if is_train:
        snn.training_stage()
    else:
        snn.load_values()
        # snn.test_accuracy(snn.get_features)
        # snn.predict_cli()
        snn.perf_test_o.perform_test(snn.predict)