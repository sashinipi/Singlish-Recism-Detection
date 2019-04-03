'''
Created on Mar 31, 2019

@author: dulan
'''

class DICTIONARY:
    SINGLISH_WORD = 'singlish'
    WORD_COUNT = 'count'

class FILES:
    TAG_DATA_FILENAME_EXCEL = ''
    TAG_DICT_FILENAME = ''
    EXCEL_DATA_FILE_PATH = ''
    DICTIONARY_FILE_PATH = ''
    CSV_FILE_PATH = '/home/dulan/learn/sashini-fyp/code/git/Singlish-Recism-Detection/data/output.csv'
    TEST_CSV_FILE_PATH = '/home/dulan/learn/sashini-fyp/code/git/Singlish-Recism-Detection/tests/test_output.csv'

class MISC:
    TAG_SAVE_COUNT = 5

class LSTM:
    LSTM_MAX_WORD_COUNT = 20
    FOLDS_COUNT = 5
    VALIDATION_TEST_SIZE = 0.2
    MAX_EPOCHS = 15
    OUTPUT_DIR = 'output'