'''
Created on Mar 31, 2019

@author: dulan
'''
from main.simple_nn import SimpleNN
from main.multinomial_db import MultinomialNBC
from main.svm import SVM

from params import SVMF, MNB
from params import PRED_TYPE

class SRD(object):
    def __init__(self):
        # load simple NN
        self.snn = SimpleNN()
        self.snn.load_values()

        self.svm = SVM()
        self.svm.load_models(SVMF)

        self.mnb = MultinomialNBC()
        self.mnb.load_models(MNB)

    def predict(self, data):
        try:
            content, type = data['content'], data['type']
            p_class, conf = "---", "---"
            if PRED_TYPE.SIMPLE_NN.value == int(type):
                p_class, conf = self.snn.predict_api(content)
            elif PRED_TYPE.SVM.value == int(type):
                p_class, conf = self.svm.predict_api(content)
            elif PRED_TYPE.MNB.value == int(type):
                p_class, conf = self.mnb.predict_api(content)


        except (ValueError)  as e:
            print(e)
            return {'Prediction': "Val-Error", 'Confidence': "0", 'Content': "--",
                    'Type': "--"}
        except KeyError:
            return {'Prediction': "Key-Error", 'Confidence': "0", 'Content': "--",
                'Type': "--"}
        else:
            return {'Prediction': str(p_class), 'Confidence': str(conf), 'Content': content,
                    'Type': PRED_TYPE(int(type)).name}


if __name__ == '__main__':
    ob = SRD()
    data = {"content": "This is test", "type": "1"}
    print(ob.predict(data))
    data = {"content": "This is test", "type": "2"}
    print(ob.predict(data))
    data = {"content": "This is test", "type": "3"}
    print(ob.predict(data))