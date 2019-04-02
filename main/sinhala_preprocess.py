'''
Created on Mar 31, 2019

@author: dulan
'''

from main.preprocess import preprocess
import string
import emoji
import numpy as np

class sinhala_preprocess(preprocess):
    LOWER_CASE = list(string.ascii_lowercase)
    UPPER_CASE = list(string.ascii_uppercase)
    LETTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    SMILES = ['😂', '😍', '🐷', '🐖', '🐽', '🔰', '🤔', '👉', '👌', '🔫', '🖕', '😇', '😈', '󾌾', '😳', '😹', '😁',
              '😤', '😡', '😔', '🏃']
    SMILES2 = ['‍♂', '💪', '😒', '😕', '😖', '😺', '❤', '💕', '😘', '💔', '😭', '😅', '😶', '🏼']
    EMOJI = list(emoji.UNICODE_EMOJI.keys())
    SPECIAL = ['…', '⁉️', ]
    PUNC = ['\"', '?', '.', '!', '(', ')', ',', '\'', '”', '“', '-', '_', '~', '=', '\\', '/', ':', '—', ' ', ' ', '󾌾',
            ' ']
    CONTAIN_STRINGS_REMOVE = PUNC + LOWER_CASE + UPPER_CASE + SPECIAL + LETTERS + EMOJI + SMILES + SMILES2
    REMOVE_WORDS_STARTING = ['@', '#', 'http']
    CONTAIN_STRINGS_REMOVE_AFTER = ['#', '@']
    LENGTH = 2

    def __init__(self):
        super(sinhala_preprocess, self).__init__()

    def pre_process(self, sentance):
        words = []
        for word in sentance.split():
            word = self.remove_letters_in_words(word, sinhala_preprocess.CONTAIN_STRINGS_REMOVE)# before
            if self.remove_words_starting(word,sinhala_preprocess.CONTAIN_STRINGS_REMOVE):
                continue
            if self.remove_by_length(word, sinhala_preprocess.LENGTH):
                continue
            word = self.remove_letters_in_words(word, sinhala_preprocess.CONTAIN_STRINGS_REMOVE_AFTER) # after
            # word = self.simplify_sinhalese_text(word) # got it from original repo
            for _word in word.split():
                _word = _word.strip()
                words.append(_word)

        words_np = np.array(words)

        return words_np