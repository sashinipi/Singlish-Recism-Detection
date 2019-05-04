'''
Created on Apr 01, 2019

@author: dulan
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC

from params import SVMF
from main.classify import classify
from main.perfomance_test import PerformanceTest

class SVM(classify):
    def __init__(self):
        super(SVM, self).__init__(SVMF.LOG_FILE_NAME)
        self.perf_test_o = PerformanceTest('SVM')

    def train(self, train_x, train_y):
        self.bow_transformer = CountVectorizer(analyzer=self.text_process).fit(train_x)
        messages_bow = self.bow_transformer.transform(train_x)
        self.tfidf_transformer = TfidfTransformer().fit(messages_bow)
        messages_tfidf = self.tfidf_transformer.transform(messages_bow)
        self.model = SVC(gamma='scale').fit(messages_tfidf, train_y)

if __name__ == '__main__':
    obj = SVM()
    is_train = False
    obj.main(is_train, SVMF)

