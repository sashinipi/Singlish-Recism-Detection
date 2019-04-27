'''
Created on Apr 23, 2019

@author: dulan
'''
import string
import emoji
from main.pickel_helper import PickelHelper

class PRE_PRO:
    pick_help = PickelHelper()

    SPECIAL = ['…', '⁉️', ]
    EMOTIONS = list(emoji.UNICODE_EMOJI.keys())
    REMOVE_WORDS_STARTING = ['@', '#', 'http']
    REMOVE_WORDS_ENDING = ['.com']
    CONTAIN_STRINGS_REMOVE_AFTER = ['#', '@']

    SINGLISH_STOP_WORDS = ['eth', 'rt', 'ekk']
    SINGLISH_SUFFIX_STRIP = ['da']
    SINGLISH_SUFFIX_REPLACE = {'a':['aa', 'aaa'], 'e':['ee', 'ee'], 'o':['oo', 'ooo']}
    SINGLISH_LEMMATIZATION = pick_help.load_obj('singlish_lemmas')
