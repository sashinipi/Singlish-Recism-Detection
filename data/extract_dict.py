'''
Created on Apr 25, 2019

@author: dulan
'''
import json
from params import PREPRO

def extract():
    filename = 'dict.json'
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

    with open(PREPRO.LEMMAS_FILENAME, 'w') as fp:
        json.dump(lemma, fp, indent=4, sort_keys=True)

if __name__ == '__main__':
    extract()