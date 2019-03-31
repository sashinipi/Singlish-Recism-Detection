'''
Created on Mar 31, 2019

@author: dulan
'''
import numpy as np
import pandas as pd
import json

from params import FILES, MISC, DICTIONARY
from main.sinhala_preprocess import sinhala_preprocess
from data.data_loader import data_loader

class tagging_tool(object):
    def __init__(self):
        self.dictionary = None
        self.sinhala_preprocess_obj = sinhala_preprocess()
        self.data_loader_obj = data_loader()

    def load_data(self):
        df = pd.read_excel(FILES.TAG_DATA_FILENAME_EXCEL)
        lines = []
        tags = []
        for line in df[df.columns[1]]:
            lines.append(line)
        for tag in df[df.columns[3]]:
            tags.append(tag)

        return np.array(lines), np.array(tags)


    def save_dict(self):
        with open(FILES.TAG_DICT_FILENAME, 'w') as fp:
            json.dump(self.dictionary, fp)

    def tag(self):
        keys_not_tagged = [key for key in self.dictionary if len(self.dictionary[key][DICTIONARY.SINGLISH_WORD]) == 0]
        total_keys_not_tagged = len(keys_not_tagged)
        total_keys_in_dict = len(self.dictionary)
        now_tag_count = 0
        for key in keys_not_tagged:
            now_tag_count += 1
            print("Tagged: {} Not-Tagged: {} Total-words: {}".format(total_keys_in_dict - total_keys_not_tagged,
                                                                     total_keys_not_tagged, total_keys_in_dict))
            self.dictionary[key][DICTIONARY.SINGLISH_WORD] = self.get_input(key)
            total_keys_not_tagged -= 1

            if now_tag_count % MISC.TAG_SAVE_COUNT == 0:
                self.save_dict()


    def get_input(self, text):
        msg = "{} : ".format(text)

        texts = []
        try:
            text = input(msg)  # Python 3
        except:
            text = raw_input(msg)  # Python 2

        for t in text.split(','):
            texts.append(t.strip())

        return texts


    def create_dict(self):
        count_save = 0
        count_duplicate = 0
        count_total_words = 0
        temp_word_list = []
        content, _ = self.load_data()
        for line in content:
            for word in self.sinhala_preprocess_obj.pre_process(line):
                count_total_words += 1
                if not (word in self.dictionary):
                    print(word)
                    self.dictionary[word] = {DICTIONARY.SINGLISH_WORD: [], DICTIONARY.WORD_COUNT: 1}
                    count_save += 1
                else:
                    count_duplicate += 1
                    if word in temp_word_list:
                        self.dictionary[word][DICTIONARY.WORD_COUNT] = self.dictionary[word][DICTIONARY.WORD_COUNT] + 1
                    else:
                        temp_word_list.append(word)
                        self.dictionary[word][DICTIONARY.WORD_COUNT] = 1

        self.save_dict()
        print("total words:{} Saved:{} Duplicate:{}".format(count_total_words, count_save, count_duplicate))

if __name__ == '__main__':
    tt = tagging_tool()
    # # TODO First time
    tt.data_loader_obj.load_dict(FILES.DICTIONARY_FILE_PATH)
    tt.create_dict()

    #TODO Tag now
    tt.data_loader_obj.load_dict(FILES.DICTIONARY_FILE_PATH)
    tt.tag()