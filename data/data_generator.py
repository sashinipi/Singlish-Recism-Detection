'''
Created on Mar 31, 2019

@author: dulan
'''
import csv
import random

from main.preprocess.preprocess import preprocess
from main.preprocess.singlish_preprocess import singlish_preprocess
from params import DICTIONARY, FILES
from data.data_loader import data_loader
from params import MISC
import logging
import numpy as np

class data_generator(object):
    def __init__(self):
        # self.ratios = ratio
        self.output_basename = 'output_{}.csv'
        self.data_loader_obj = data_loader()
        self.dictionary = self.data_loader_obj.load_dict(FILES.DICTIONARY_FILE_PATH)
        self.pre_process_o = singlish_preprocess()
        # self.convert_to_singlish()
        self.type_count = None

    def write_to_csv(self, write_this_list, type=''):
        with open(self.output_basename.format(type), 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(write_this_list)
        writeFile.close()

    def get_random_word(self, list_of_words):
        length_of_list = len(list_of_words)
        if length_of_list == 1:
            return list_of_words[0]
        else:
            return list_of_words[random.randint(0, length_of_list - 1)]

    def set_count(self, split_dict, tags):

        totals = [np.count_nonzero(tags == MISC.CLASSES[0]), np.count_nonzero(tags == MISC.CLASSES[1])]
        self.type_count = {}
        self.type_limit = {}
        for type in split_dict.keys():
            self.type_count[type] = {}
            for i in range(2):
                self.type_count[type][MISC.CLASSES[i]] = {}
                self.type_count[type][MISC.CLASSES[i]]['limit'] = int(float(split_dict[type]) * int(totals[i]))
                self.type_count[type][MISC.CLASSES[i]]['count'] = 0

        logging.info(self.type_limit)

    def get_type(self, clas):
        for key in self.type_count.keys():
            if self.type_count[key][clas]['count'] < self.type_count[key][clas]['limit']:
                self.type_count[key][clas]['count'] += 1
                return key

    def convert_to_singlish(self, split_dict = None, augment=1):
        content, tags = self.data_loader_obj.load_data_from_excel(FILES.EXCEL_DATA_FILE_PATH)
        self.set_count(split_dict, tags)
        for i, line in enumerate(content):
            words = self.pre_process_o.pre_process(line)
            _type = self.get_type(tags[i])
            for j in range(augment):
                for word in words:
                    if word in self.dictionary.keys():
                        if not self.dictionary[word][DICTIONARY.SINGLISH_WORD] == []:
                            line = line.replace(word, self.get_random_word(self.dictionary[word][DICTIONARY.SINGLISH_WORD]))

                self.write_to_csv([line, tags[i]],type=_type)
            logging.debug(line, tags[i])
        logging.info(self.type_count)

if __name__ == '__main__':
    dg_obj = data_generator()
    dg_obj.convert_to_singlish(split_dict = {'train':0.8, 'test':0.2}, augment=2)
