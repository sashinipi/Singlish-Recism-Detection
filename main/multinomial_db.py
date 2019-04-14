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
        super(MultinomialNBC, self).__init__()

    def train(self, train_x, train_y):
        bow_transformer = CountVectorizer(analyzer=self.text_process).fit(train_x)
        messages_bow = bow_transformer.transform(train_x)
        tfidf_transformer = TfidfTransformer().fit(messages_bow)

        messages_tfidf = tfidf_transformer.transform(messages_bow)
        spam_detect_model = MultinomialNB().fit(messages_tfidf, train_y)

        return spam_detect_model, bow_transformer, tfidf_transformer

if __name__ == '__main__':
    MultinomialNB_obj = MultinomialNBC()
    is_train = True
    if is_train:
        MultinomialNB_obj.train_test()
        MultinomialNB_obj.save_models(MNB)
    else:
        MultinomialNB_obj.load_models(MNB)
