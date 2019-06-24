'''
Created on Apr 01, 2019

@author: dulan
'''
from nltk.corpus import stopwords
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from params import FILES

class MultinomialNBC(object):
    def __init__(self):
        pass

def text_process(mess):
    # nopunc = [char for char in mess if char not in string.punctuation]
    # nopunc = ''.join(nopunc)
    # return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return mess#self.singlish_preprocess_obj.pre_process(mess)

def split_data():
    messages = pd.read_csv(FILES.CSV_FILE_PATH, sep=',', names=["message", "label"])
    print(messages.head())
    msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

    print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

    return msg_train, msg_test, label_train, label_test

def get_report(classifier, msg_train, msg_test, label_train, label_test):
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', classifier),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])

    pipeline.fit(msg_train, label_train)
    predictions = pipeline.predict(msg_test)
    print(classification_report(predictions, label_test))

def get_fet(msg_train):
    bow_transformer = CountVectorizer(analyzer=text_process).fit(msg_train)
    messages_bow = bow_transformer.transform(msg_train)
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)

    # Print total number of vocab words
    # print("Length of feature vec :", len(bow_transformer.vocabulary_))
    # bw_msg = bow_transformer.transform(msg_train)
    # tfidef_msg = tfidf_transformer.transform(bw_msg)
    #
    # print(bw_msg.shape)
    # print(tfidef_msg.shape)

    return messages_tfidf

def trans_val(val):
    if val == 'ham':
        return [1,0]
    else:
        return [0,1]

def main():
    msg_train, msg_test, label_train, label_test = split_data()
    print("=========== MultinomialNB ============")
    mul = MultinomialNB()
    get_report(mul, msg_train, msg_test, label_train, label_test)
    print ("=========== SVC ============")
    get_report(SVC(gamma='scale'), msg_train, msg_test, label_train, label_test)

main()
