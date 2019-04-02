'''
Created on Apr 01, 2019

@author: dulan
'''

class model(object):
    def __init__(self):
        pass

    def create_model(self):
        raise NotImplementedError

    def train(self, x_corpus, y_corpus):
        raise NotImplementedError

    def predict(self):
        pass
