'''
Created on Mar 31, 2019

@author: dulan
'''
import numpy as np
import string
import emoji

from main.preprocess.preprocess import preprocess

class singlish_preprocess(preprocess):
    LOWER_CASE = list(string.ascii_lowercase)
    UPPER_CASE = list(string.ascii_uppercase)
    NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    SMILES = ['😂', '😍', '🐷', '🐖', '🐽', '🔰', '🤔', '👉', '👌', '🔫', '🖕', '😇', '😈', '󾌾', '😳', '😹', '😁',
              '😤', '😡', '😔', '🏃']
    SMILES2 = ['‍♂', '💪', '😒', '😕', '😖', '😺', '❤', '💕', '😘', '💔', '😭', '😅', '😶', '🏼']
    EMOJI = list(emoji.UNICODE_EMOJI.keys())
    SPECIAL = ['…', '⁉️', ]
    PUNC = ['\"', '?', '.', '!', '(', ')', ',', '\'', '”', '“', '-', '_', '~', '=', '\\', '/', ':', '—', ' ', ' ', '󾌾',
            ' ']
    CONTAIN_STRINGS_REMOVE = PUNC + SPECIAL + NUMBERS
    EMOTIONS = EMOJI + SMILES + SMILES2
    REMOVE_WORDS_STARTING = ['@', '#', 'http']
    CONTAIN_STRINGS_REMOVE_AFTER = ['#', '@']
    LENGTH = 2

    def __init__(self):
        super(singlish_preprocess, self).__init__()

    def add_spaces_for_emojis(self, word, emojis):
        for emo in emojis:
            if emo in word:
                word = word.replace(emo, ' '+emo+' ')
        return word

    def pre_process(self, sentence):
        words = []
        for word in sentence.split():
            word = self.remove_letters_in_words(word, singlish_preprocess.CONTAIN_STRINGS_REMOVE)  # before
            if self.remove_words_starting(word, singlish_preprocess.REMOVE_WORDS_STARTING):
                continue
            if self.remove_by_length(word, singlish_preprocess.LENGTH):
                continue
            word = self.remove_letters_in_words(word, singlish_preprocess.CONTAIN_STRINGS_REMOVE_AFTER)  # after
            # word = self.simplify_sinhalese_text(word) # got it from original repo
            word = self.add_spaces_for_emojis(word, singlish_preprocess.EMOTIONS)
            for _word in word.split():
                _word = _word.strip()
                words.append(_word)

        words_np = np.array(words)

        return words_np