'''
Created on Mar 31, 2019

@author: dulan
'''
from main.simple_nn import SimpleNN

from params import PRED_TYPE

class SRD(object):
    def __init__(self):
        # load simple NN
        self.snn = SimpleNN()
        self.snn.load_values()

    def predict(self, data):
        try:
            content, type = data['content'], data['type']
            p_class, conf = "---", "---"
            if PRED_TYPE.SIMPLE_NN.value == int(type):
                p_class, conf = self.snn.predict_api(content)

        except (ValueError):
            return {'Prediction': "Val-Error", 'Confidence': "0", 'Content': "--",
                    'Type': "--"}
        except KeyError:
            return {'Prediction': "Key-Error", 'Confidence': "0", 'Content': "--",
                'Type': "--"}
        else:
            return {'Prediction': str(p_class), 'Confidence': str(conf), 'Content': content,
                    'Type': PRED_TYPE(int(type)).name}


