'''
Created on Apr 23, 2019

@author: dulan
'''
import string
import emoji

class PRE_PRO:
    # LOWER_CASE = list(string.ascii_lowercase)
    # UPPER_CASE = list(string.ascii_uppercase)
    #
    # SMILES = ['😂', '😍', '🐷', '🐖', '🐽', '🔰', '🤔', '👉', '👌', '🔫', '🖕', '😇', '😈', '󾌾', '😳', '😹', '😁',
    #           '😤', '😡', '😔', '🏃']
    # SMILES2 = ['‍♂', '💪', '😒', '😕', '😖', '😺', '❤', '💕', '😘', '💔', '😭', '😅', '😶', '🏼']
    # EMOJI = list(emoji.UNICODE_EMOJI.keys())

    SPECIAL = ['…', '⁉️', ]
    # PUNC = ['\"', '?', '.', '!', '(', ')', ',', '\'', '”', '“', '-', '_', '~', '=', '\\', '/', ':', '—', ' ', ' ', '󾌾',
    #         ' ']
    # NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #
    # CONTAIN_STRINGS_REMOVE = PUNC + SPECIAL
    EMOTIONS = list(emoji.UNICODE_EMOJI.keys())
    REMOVE_WORDS_STARTING = ['@', '#', 'http']
    CONTAIN_STRINGS_REMOVE_AFTER = ['#', '@']

    SINGLISH_STOP_WORDS = ['eth', 'RT']
    SINGLISH_SUFFIX_STRIP = ['da']
    SINGLISH_SUFFIX_REPLACE = {'a':['aa', 'aaa'], 'e':['ee', 'ee'], 'o':['oo', 'ooo']}
    SINGLISH_LEMMATIZATION = {'mama': ['mame', 'man'], 'oya':['oa', 'oy', 'ohe'], 'uta':['uuta']}
