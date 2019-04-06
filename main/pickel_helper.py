'''
Created on Apr 06, 2019

@author: dulan
'''
import os
import pickle
import os.path as osp

class PickelHelper(object):
    DEF_SAV_LOC = 'save'
    def __init__(self):
        if not osp.exists(PickelHelper.DEF_SAV_LOC):
            os.makedirs(PickelHelper.DEF_SAV_LOC)

    def save_obj(self, name, obj):
        with open(osp.join(PickelHelper.DEF_SAV_LOC, name+'.pkl'), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        with open(osp.join(PickelHelper.DEF_SAV_LOC, name+'.pkl'), 'rb') as f:
            return pickle.load(f)