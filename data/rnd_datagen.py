'''
Created on May 19, 2019

@author: dulan
'''
import random
import numpy as np
import csv

class RnDDatagen(object):
    def __init__(self):
        r_data_n = ['thambiya', 'popsiya', 'nana']
        n_data_n = ['honda', 'hamoma', 'ekata']
        self.data = {"Racist":{"first":['maranna', 'kapanna'], "second": ['oni', 'epa'], "normal" : r_data_n}
            ,"Neutral": {"first":['oni', 'epa'], "second": ['maranna', 'kapanna'], "normal": n_data_n}}
        self.output_basename = "output_train_rnd.csv"

    def write_to_csv(self, write_this_list, type=''):
        with open(self.output_basename.format(type), 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(write_this_list)
        writeFile.close()

    def create_train(self):

        for d in ["Racist", "Neutral"]:
            for i in range(1500):
                sen = np.array([])
                start = random.randint(1,6)
                sen_len = random.randint(start,start+2)
                for j in range(sen_len):
                    word_place = random.randint(0, len(self.data[d]['normal'])-1)
                    word = self.data[d]['normal'][word_place]+ " "
                    sen = np.append(sen, word)

                first_place = random.randint(0,sen_len-1)
                sen = np.insert(sen, first_place,
                                self.data[d]['first'][random.randint(0, len(self.data[d]['first']) - 1)] + " ")
                second_place = random.randint(first_place+1,sen_len)
                # print("{} - {} - {}".format(sen, first_place, second_place))

                sen = np.insert(sen, second_place, self.data[d]['second'][random.randint(0, len(self.data[d]['second'])-1)] + " ")
                print("{},{}".format( "".join(sen), d))
                self.write_to_csv(["".join(sen), d])


if __name__ == '__main__':
    obj = RnDDatagen()
    obj.create_train()