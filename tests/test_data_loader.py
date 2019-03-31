'''
Created on Mar 31, 2019

@author: dulan
'''
from unittest import TestCase


class TestData_loader(TestCase):
    def test_load_data(self):
        from data.data_loader import data_loader
        filename = 'test_output.csv'
        dl = data_loader(filename)
        x,y=dl.load_data()
        self.assertTrue(y[0]=='Racist')
        self.assertEqual(x[0], 'eyala aththatma oyawa maranna hadankota islamphobiyawak nmei.')
