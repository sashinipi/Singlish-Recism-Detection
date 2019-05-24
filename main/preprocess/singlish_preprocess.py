'''
Created on Mar 31, 2019

@author: dulan
'''
import numpy as np
from main.preprocess.list_of_words import PRE_PRO
import re
from main.preprocess.preprocess import preprocess
import logging

class singlish_preprocess(preprocess):
    LENGTH = 2

    def __init__(self):
        super(singlish_preprocess, self).__init__()
        # logging.basicConfig(level=logging.INFO)

    def remove_stop_words(self, word):
        return word in PRE_PRO.SINGLISH_STOP_WORDS

    def pre_process(self, sentence):
        # logging.debug('Before pre-processing:'+ sentence)

        processed_sentence = ''
        for word in sentence.split():
            # remove words starting from this letters
            if self.is_words_starting(word, PRE_PRO.REMOVE_WORDS_STARTING) is not None:
                continue
            # remove words starting from this letters
            if self.is_words_ending(word, PRE_PRO.REMOVE_WORDS_ENDING) is not None:
                continue
            # convert to lowercase
            word = self.convert_to_lowercase(word)
            #remove punctuations
            word = self.remove_punc(word)
            #Remove special charactors
            word = self.remove_letters_in_words(word, PRE_PRO.SPECIAL)
            # remove numbers
            word = self.remove_numbers(word)
            # word = self.remove_letters_in_words(word, PRE_PRO.NUMBERS)
            # Stripping suffixes
            word = self.suffix_stripping(word, PRE_PRO.SINGLISH_SUFFIX_STRIP)
            #replacing suffix
            word = self.suffix_replace(word, PRE_PRO.SINGLISH_SUFFIX_REPLACE)
            # remove lemmatization words
            word = self.lemmatization(word, PRE_PRO.SINGLISH_LEMMATIZATION)
            #Remove stop words
            if self.remove_stop_words(word):
                continue

            #Remove words that are less than this length
            if self.remove_by_length(word, singlish_preprocess.LENGTH):
                continue
            word = self.remove_letters_in_words(word, PRE_PRO.CONTAIN_STRINGS_REMOVE_AFTER)  # after
            # word = self.simplify_sinhalese_text(word) # got it from original repo

            #remove emojies
            word = self.add_spaces_for_emojis(word, PRE_PRO.EMOTIONS)
            # create the list of words

            for _word in word.split():
                _word = _word.strip()
                processed_sentence += _word + ' '

        # logging.debug('After preprocessing:')
        # logging.debug(words)
        # Create a numpy and send

        processed_sentence = re.sub(r'[^a-zA-Z0-9_]', ' ', processed_sentence)
        words = []
        for _word in processed_sentence.split():
            _word = _word.strip()
            words.append(_word)

        words_np = np.array(words)

        return words_np

if __name__ == '__main__':
    singlish_preprocess_obj = singlish_preprocess()
    print(singlish_preprocess_obj.pre_process('islam ammadya .. .. meh kattiya eksath rajadaniye therak nopawathinna #TooManyPakis #GTFO karanne.""" ™ ðÿ ™ '))