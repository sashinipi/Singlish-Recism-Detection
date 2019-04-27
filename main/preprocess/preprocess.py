'''
Created on Mar 31, 2019

@author: dulan
'''
import logging
import string
import emoji

class preprocess(object):
    def __init__(self):
        logging.basicConfig(level=logging.INFO)

    def convert_to_lowercase(self, word):
        return word.lower()

    def remove_punc(self, word):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in word if ch not in exclude)

    def remove_numbers(self, word):
        return ''.join(ch for ch in word if not ch.isdigit())

    def remove_emojis(self, sentence):
        str_sen = str(sentence)
        EMO = list(emoji.UNICODE_EMOJI.keys())
        for ch in str_sen:
            if ch in EMO:
                str_sen.replace(ch, ' ')
        return str_sen

    def add_spaces_for_emojis(self, word, emojis):
        for emo in emojis:
            if emo in word:
                word = word.replace(emo, ' '+emo+' ')
        return word

    def is_words_starting(self, word, starting_words):
        ret = None
        for let_part in starting_words:
            if word[0:len(let_part)] == let_part:
                logging.debug('Word is starting : {} Word: {}'.format(let_part, word))
                ret = let_part
                break
        return ret

    def is_words_ending(self, word, ending_words):
        ret = None
        for let_part in ending_words:
            if word[-len(let_part):] == let_part:
                ret = let_part
                break
        return ret

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

    def suffix_replace(self, word, suffixes):
        for key in suffixes.keys():
            part = self.is_words_ending(word, suffixes[key])
            if part is not None:
                logging.debug('suffix replace input:'+ word)
                ret = word[:-len(part)]+key
                logging.debug('suffix replace output:'+ret)
                return ret
        return word

    def suffix_stripping(self, word, suffixes):
        part = self.is_words_ending(word, suffixes)
        if part is not None:
            logging.debug('suffix {} stripping input:{}'.format(part, word))
            ret = word[:-len(part)]
            logging.debug('suffix stripping output:' +ret)
            return ret
        return word

    def lemmatization(self, word, lemmatization_words):
        for key in lemmatization_words.keys():
            if word in lemmatization_words[key]:
                logging.debug('Lemmatized word: {} to {}'.format(word, key))
                return key
        return word

    def pre_process(self, sentence):
        raise NotImplementedError