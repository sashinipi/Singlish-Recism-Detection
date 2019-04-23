'''
Created on Apr 01, 2019

@author: dulan
'''
from params import DIR, MISC
import os.path as osp
from main.pickel_helper import PickelHelper
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from main.classify import classify

class Model(object):
    def __init__(self):
        self.model = None  # self.create_model(input_dim)
        self.bow_transformer = None
        self.tfidf_transformer = None
        self.pic_obj = PickelHelper()
        self.classify = classify()

    def create_model(self, input_dim):
        raise NotImplementedError

    def train(self, x_corpus, y_corpus):
        raise NotImplementedError

    def train_with_test(self, x_corpus, y_corpus, x_corpus_, y_corpus_):
        raise NotImplementedError

    def predict(self, text):
        raise NotImplementedError

    def save_model(self, model_name=''):
        self.model.save_weights(osp.join(DIR.DEF_SAV_LOC, model_name))

    def load_model(self, weights_file=''):
        self.model.load_weights(osp.join(DIR.DEF_SAV_LOC, weights_file))

    def save_transformers(self, filename):
        transform = {'bow': self.bow_transformer, 'tfidf': self.tfidf_transformer}
        self.pic_obj.save_obj(filename, transform)

    def load_transformers(self, filename):
        transform = self.pic_obj.load_obj(filename)
        self.tfidf_transformer = transform['tfidf']
        self.bow_transformer = transform['bow']

    def train_feature_gen(self, msg_train):
        bow_transformer = CountVectorizer(analyzer=self.classify.text_process).fit(msg_train)
        messages_bow = bow_transformer.transform(msg_train)
        tfidf_transformer = TfidfTransformer().fit(messages_bow)
        messages_tfidf = tfidf_transformer.transform(messages_bow)

        self.bow_transformer = bow_transformer
        self.tfidf_transformer = tfidf_transformer
        return messages_tfidf

    def trans_val(self, val):
        if val == MISC.CLASSES[0]:
            return [1, 0]
        else:
            return [0, 1]

    def get_features(self, data):
        bw_msg = self.bow_transformer.transform(data)
        tfidef_msg = self.tfidf_transformer.transform(bw_msg)
        return tfidef_msg

    def evaluate(self, X, Y):
        # evaluate the model
        scores = self.model.evaluate(X, Y)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
        return scores[1]