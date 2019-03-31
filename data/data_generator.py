'''
Created on Mar 31, 2019

@author: dulan
'''
import csv
import random

from main.preprocess import preprocess
from params import DICTIONARY, FILES
from data.data_loader import data_loader


class data_generator(object):
    def __init__(self, ratio):
        self.ratios = ratio
        self.output_basename = 'output.csv'
        self.data_loader_obj = data_loader()
        self.dictionary = self.data_loader_obj.load_dict(FILES.DICTIONARY_FILE_PATH)
        self.pre_process_o = preprocess()
        self.convert_to_singlish()

    def write_to_csv(self, write_this_list):
        with open(self.output_basename, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(write_this_list)
        writeFile.close()

    def get_random_word(self, list_of_words):
        length_of_list = len(list_of_words)
        if length_of_list == 1:
            return list_of_words[0]
        else:
            return list_of_words[random.randint(0, length_of_list - 1)]

    def convert_to_singlish(self):
        content, tags = self.data_loader_obj.load_data_from_excel(FILES.EXCEL_DATA_FILE_PATH)
        for i, line in enumerate(content):
            print(line)
            words = self.pre_process_o.pre_process(line)
            for word in words:
                if word in self.dictionary.keys():
                    if not self.dictionary[word][DICTIONARY.SINGLISH_WORD] == []:
                        line = line.replace(word, self.get_random_word(self.dictionary[word][DICTIONARY.SINGLISH_WORD]))

            print(line, tags[i])
            self.write_to_csv([line, tags[i]])

