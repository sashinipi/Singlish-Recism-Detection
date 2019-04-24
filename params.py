'''
Created on Mar 31, 2019

@author: dulan
'''
from enum import Enum, unique
import os
import os.path as osp

ABS_PATH_TO_FILE = os.path.dirname(__file__)

@unique
class PRED_TYPE(Enum):
    SIMPLE_NN   = 1
    SVM         = 2
    MNB         = 3

class DICTIONARY:
    SINGLISH_WORD   = 'singlish'
    WORD_COUNT      = 'count'

class DIR:
    DEF_SAV_LOC     = osp.join(ABS_PATH_TO_FILE, 'main/save')
    LOGS_DIR        = osp.join(ABS_PATH_TO_FILE, 'main/logs')

class FILES:
    TAG_DATA_FILENAME_EXCEL = ''
    TAG_DICT_FILENAME = ''
    EXCEL_DATA_FILE_PATH    = osp.join(ABS_PATH_TO_FILE, 'data/final-data-set.xlsx')
    DICTIONARY_FILE_PATH    = osp.join(ABS_PATH_TO_FILE, 'data/dict.json')
    CSV_FILE_PATH           = osp.join(ABS_PATH_TO_FILE, 'data/output.csv')
    SEP_CSV_FILE_PATHS      = osp.join(ABS_PATH_TO_FILE, 'data/output_{}.csv')
    TEST_CSV_FILE_PATH      = osp.join(ABS_PATH_TO_FILE, 'tests/test_output.csv')

class MISC:
    TAG_SAVE_COUNT = 5
    CLASSES = ['Racist', 'Neutral']

class LSTMP:
    LSTM_MAX_WORD_COUNT     = 60
    FOLDS_COUNT             = 4
    VALIDATION_TEST_SIZE    = 0.3
    MAX_EPOCHS              = 8
    BATCH_SIZE              = 4
    OUTPUT_DIR              = 'save'
    DATA_SET_CLASSES = {
        'Neutral': [0, 1],
        'Racist': [1, 0]
    }

    DICTIONARY_FILENAME = 'lstm_dict'
    MODEL_FILENAME = 'lstm_model.h5'
    MODEL_FILENAME_ACC = 'lstm_model_{}.h5'
    LOG_FILE_NAME = 'lstm-model'


class SNN:
    TRANS_FILENAME = 'snn_trans'
    MODEL_FILENAME = 'snn_model.h5'
    INPUT_FILENAME = 'snn_input'


class NN:
    FOLDS_COUNT             = 5
    VALIDATION_TEST_SIZE    = 0.2
    MAX_EPOCHS              = 10
    OUTPUT_DIR              = 'save'
    DATA_SET_CLASSES = {
        'Neutral': [0, 1],
        'Racist': [1, 0]
    }
    TRANS_FILENAME = 'nn_trans'
    MODEL_FILENAME = 'nn_model.h5'
    INPUT_FILENAME = 'nn_input'

class SVMF():
    BOW_FILENAME    = 'svm_bow'
    TFIDF_FILENAME  = 'svm_tfidf'
    MODEL_FILENAME  = 'svm_model'
    INPUT_FILENAME  = 'svm_input'

class MNB():
    BOW_FILENAME    = 'mnb_bow'
    TFIDF_FILENAME  = 'mnb_tfidf'
    MODEL_FILENAME  = 'mnb_model'
    INPUT_FILENAME  = 'mnb_input'