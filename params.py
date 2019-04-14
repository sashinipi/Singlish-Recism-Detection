'''
Created on Mar 31, 2019

@author: dulan
'''
from enum import Enum, unique
@unique
class PRED_TYPE(Enum):
    SIMPLE_NN = 1
    SVM = 2
    MNB = 3

class DICTIONARY:
    SINGLISH_WORD = 'singlish'
    WORD_COUNT = 'count'

class DIR:
    DEF_SAV_LOC = 'save'
    LOGS_DIR = 'logs'

class FILES:
    TAG_DATA_FILENAME_EXCEL = ''
    TAG_DICT_FILENAME = ''
    EXCEL_DATA_FILE_PATH = '/home/dulan/learn/sashini-fyp/code/git/Singlish-Recism-Detection/data/final-data-set.xlsx'
    DICTIONARY_FILE_PATH = '/home/dulan/learn/sashini-fyp/code/git/Singlish-Recism-Detection/data/dict.json'
    CSV_FILE_PATH = '/home/dulan/learn/sashini-fyp/code/git/Singlish-Recism-Detection/data/output.csv'
    SEP_CSV_FILE_PATHS = '/home/dulan/learn/sashini-fyp/code/git/Singlish-Recism-Detection/data/output_{}.csv'
    TEST_CSV_FILE_PATH = '/home/dulan/learn/sashini-fyp/code/git/Singlish-Recism-Detection/tests/test_output.csv'

class MISC:
    TAG_SAVE_COUNT = 5
    CLASSES = ['Racist', 'Neutral']

class LSTMP:
    LSTM_MAX_WORD_COUNT = 60
    FOLDS_COUNT = 5
    VALIDATION_TEST_SIZE = 0.2
    MAX_EPOCHS = 20
    OUTPUT_DIR = 'save'
    DATA_SET_CLASSES = {
        'Neutral': [0, 1],
        'Racist': [1, 0]
    }

class SVMF():
    BOW_FILENAME = 'svm_bow'
    TFIDF_FILENAME = 'svm_tfidf'
    MODEL_FILENAME = 'svm_model'
    INPUT_FILENAME = 'svm_input'

class MNB():
    BOW_FILENAME = 'mnb_bow'
    TFIDF_FILENAME = 'mnb_tfidf'
    MODEL_FILENAME = 'mnb_model'
    INPUT_FILENAME = 'mnb_input'