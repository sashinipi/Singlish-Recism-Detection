'''
Created on Mar 31, 2019

@author: dulan
'''
import numpy as np

from srd.preprocess import preprocess

class singlish_preprocess(preprocess):

    def __init__(self):
        super(singlish_preprocess, self).__init__()

    def pre_process(self, sentance):
        words = []
        for word in sentance.split():
            # print("pre-preocess:", word)

            word = self.remove_letters_in_words(word, remove_string)# before
            if self.remove_words_starting(word,starting_words):
                continue
            if self.remove_by_length(word, length):
                continue
            word = self.remove_letters_in_words(word, remove_string) # after
            # word = self.simplify_sinhalese_text(word) # got it from original repo
            for _word in word.split():
                _word = _word.strip()
                words.append(_word)

        words_np = np.array(words)

        return words_np