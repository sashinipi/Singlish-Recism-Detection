'''
Created on Apr 23, 2019

@author: dulan
'''
import string
import emoji
import json
from params import PREPRO
from main.pickel_helper import JsonHelper

class PRE_PRO():

    jh_obj = JsonHelper()
    SPECIAL = ['…', '⁉️', ]
    EMOTIONS = list(emoji.UNICODE_EMOJI.keys())
    REMOVE_WORDS_STARTING = ['@', '#', 'http']
    REMOVE_WORDS_ENDING = ['.com']
    CONTAIN_STRINGS_REMOVE_AFTER = ['#', '@']

    SINGLISH_STOP_WORDS = ['eth', 'rt', 'ekk']
    SINGLISH_SUFFIX_STRIP = ['da']
    SINGLISH_SUFFIX_REPLACE = {'a':['aa', 'aaa'], 'e':['ee', 'eee'], 'o':['oo', 'ooo']}
    SINGLISH_LEMMATIZATION = jh_obj.load_json(PREPRO.LEMMAS_FILENAME)
