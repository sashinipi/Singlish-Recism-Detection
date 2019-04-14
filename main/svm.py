'''
Created on Apr 01, 2019

@author: dulan
'''
import nltk
from nltk.corpus import stopwords
import pandas as pd
# import seaborn as sns
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from params import SVMF

from main.classify import classify

class SVM(classify):
    def __init__(self):
        super(SVM, self).__init__()

    def train(self, train_x, train_y):
        self.bow_transformer = CountVectorizer(analyzer=self.text_process).fit(train_x)
        messages_bow = self.bow_transformer.transform(train_x)
        self.tfidf_transformer = TfidfTransformer().fit(messages_bow)
        messages_tfidf = self.tfidf_transformer.transform(messages_bow)
        self.model = MultinomialNB().fit(messages_tfidf, train_y)

if __name__ == '__main__':
    svm_obj = SVM()
    is_train = True
    if is_train:
        svm_obj.train_test()
        svm_obj.save_models(SVMF)
    else:
        svm_obj.load_models(SVMF)
