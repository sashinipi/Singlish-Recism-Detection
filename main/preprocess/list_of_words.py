'''
Created on Apr 23, 2019

@author: dulan
'''
import string
import emoji
import json
from params import PREPRO

class PRE_PRO():
    SPECIAL = ['…', '⁉️', ]
    EMOTIONS = list(emoji.UNICODE_EMOJI.keys())
    REMOVE_WORDS_STARTING = ['@', '#', 'http']
    REMOVE_WORDS_ENDING = ['.com']
    CONTAIN_STRINGS_REMOVE_AFTER = ['#', '@']

    SINGLISH_STOP_WORDS = ['eth', 'rt', 'ekk']
    SINGLISH_SUFFIX_STRIP = ['da']
    SINGLISH_SUFFIX_REPLACE = {'a':['aa', 'aaa'], 'e':['ee', 'eee'], 'o':['oo', 'ooo']}

    @staticmethod
    def load_lemmas():
        with open(PREPRO.LEMMAS_FILENAME, 'r') as fp:
            ret = json.load(fp)
        return ret

    SINGLISH_LEMMATIZATION = load_lemmas()
