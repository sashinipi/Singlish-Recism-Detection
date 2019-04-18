'''
Created on Apr 06, 2019

@author: dulan
'''
import os
import pickle
import os.path as osp
from params import DIR


class PickelHelper(object):
    def __init__(self):
        if not osp.exists(DIR.DEF_SAV_LOC):
            os.makedirs(DIR.DEF_SAV_LOC)

    def save_obj(self, name, obj):
        with open(osp.join(DIR.DEF_SAV_LOC, name+'.pkl'), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        with open(osp.join(DIR.DEF_SAV_LOC, name+'.pkl'), 'rb') as f:
            return pickle.load(f)