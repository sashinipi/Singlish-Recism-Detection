'''
Created on Apr 25, 2019

@author: dulan
'''
import json
from main.pickel_helper import PickelHelper
def extract():
    filename = 'dict.json'
    pick_help = PickelHelper()
    with open(filename, 'r') as fp:
        dict = json.load(fp)

    count = 0
    lemma = {}
    for key in dict:
        sinlgish_list = dict[key]['singlish']
        if len(sinlgish_list) > 1:
            lemma[sinlgish_list[0]] = sinlgish_list[1:]
            count += 1

    print("Total : {}".format(count))
    print(lemma)
    pick_help.save_obj('singlish_lemmas', lemma)

if __name__ == '__main__':
    extract()