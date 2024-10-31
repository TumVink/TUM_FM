'''
Merge two process.csv files of two datasets generated from CLAM pacage.
Jingsong Liu
'''

import pandas as pd
from os.path import join

def Merge(csv1, csv2):
    '''
    Merge two csv files of two datasets generated from CLAM pacage.
    :param csv1: process1.csv
    :param csv2: process2.csv
    :return: dataset_new.csv
    '''

    #read csv1 and csv2
    dataset_1 = pd.read_csv(csv1)
    dataset_2= pd.read_csv(csv2)




