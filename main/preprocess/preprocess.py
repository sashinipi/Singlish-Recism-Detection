'''
Created on Mar 31, 2019

@author: dulan
'''

class preprocess(object):
    def __init__(self):
        pass

    def convert_to_lowercase(self, word):
        return word.lower()

    def remove_words_starting(self, word, starting_words):
        flag = False
        for let_part in starting_words:
            if word[0:len(let_part)] == let_part:
                flag = True
                break
        return flag

    def remove_by_length(self, word, length):
        flag = False
        if len(word) < length:
            flag = True
        return flag

    def remove_letters_in_words(self, word, remove_string):
        for let in remove_string:
            if let in word:
                word = word.replace(let, ' ')
        return word

    def pre_process(self, sentance):
        raise NotImplementedError