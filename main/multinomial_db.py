'''
Created on Apr 01, 2019

@author: dulan
'''
from main.classify import classify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from params import MNB


class MultinomialNBC(classify):
    def __init__(self):
        super(MultinomialNBC, self).__init__(MNB.LOG_FILE_NAME)

    def train(self, train_x, train_y):
        self.bow_transformer = CountVectorizer(analyzer=self.text_process).fit(train_x)
        messages_bow = self.bow_transformer.transform(train_x)
        self.tfidf_transformer = TfidfTransformer().fit(messages_bow)
        self.model = MultinomialNB().fit(self.tfidf_transformer.transform(messages_bow), train_y)

if __name__ == '__main__':
    MultinomialNB_obj = MultinomialNBC()
    is_train = True
    MultinomialNB_obj.main(is_train, MNB)
