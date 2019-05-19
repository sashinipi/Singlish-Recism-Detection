'''
Created on May 19, 2019

@author: dulan
'''
from main.lstm_model import LSTMModel
from keras import regularizers
from keras.layers import Dense, GRU
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import Adam

from params import GRUP


class GRU_Model(LSTMModel):
    def __init__(self):
        super(GRU_Model, self).__init__()

    def create_model(self, input_dim):
        # ################## Deep Neural Network Model ###################### #
        print("=========== Loading GRU Model ===============")
        model = Sequential()
        model.add(Embedding(input_dim=input_dim, output_dim=60, input_length=GRUP.LSTM_MAX_WORD_COUNT))
        model.add(GRU(units=600))
        model.add(Dense(units=GRUP.LSTM_MAX_WORD_COUNT, activation='tanh', kernel_regularizer=regularizers.l2(0.04),
                        activity_regularizer=regularizers.l2(0.015)))
        model.add(Dense(units=GRUP.LSTM_MAX_WORD_COUNT, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                        bias_regularizer=regularizers.l2(0.01)))
        model.add(Dense(units=2, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))
        adam_optimizer = Adam(lr=0.001, decay=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

        self.logger.info("Model Architecture")
        self.logger.info(model.summary())
        return model

if __name__ == '__main__':
    gru_obj = GRU_Model()
    train = False
    if train:
        gru_obj.train_now()
        gru_obj.predict_cli()
    else:
        gru_obj.load_values()
        gru_obj.predict_cli()