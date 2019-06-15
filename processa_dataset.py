# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:42:43 2019
"""
from os import listdir
from os.path import isfile, join

import pandas as pd

import numpy as np

import h5py

wnd = 500
sensors = 8
dataset_train = []
label_train = []

dataset_test = []
label_test = []

for i in range(1,2):
    mypath = 'dataset/' + str(i)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    filepath = 'dataset/'+ str(i) + '/' + onlyfiles[0]
    aux_data = pd.read_csv(filepath, delimiter = '\t')

    ls_classes = np.unique(aux_data['class'])
    
    print(filepath)
    
    for j in range(1,len(ls_classes)):
        channel1 = aux_data[aux_data['class'] == ls_classes[j]]['channel1']
        channel2 = aux_data[aux_data['class'] == ls_classes[j]]['channel2']
        channel3 = aux_data[aux_data['class'] == ls_classes[j]]['channel3']
        channel4 = aux_data[aux_data['class'] == ls_classes[j]]['channel4']
        channel5 = aux_data[aux_data['class'] == ls_classes[j]]['channel5']
        channel6 = aux_data[aux_data['class'] == ls_classes[j]]['channel6']
        channel7 = aux_data[aux_data['class'] == ls_classes[j]]['channel7']
        channel8 = aux_data[aux_data['class'] == ls_classes[j]]['channel8']
        
        for k in range(len(channel1) - wnd):
            ch_A = channel1[k : k + wnd]
            ch_B = channel2[k : k + wnd]
            ch_C = channel3[k : k + wnd]
            ch_D = channel4[k : k + wnd]
            ch_E = channel5[k : k + wnd]
            ch_F = channel6[k : k + wnd]
            ch_G = channel7[k : k + wnd]
            ch_H = channel8[k : k + wnd]
            
            dataset_train.append([ch_A, ch_B, ch_C, ch_D,\
                            ch_E, ch_F, ch_G, ch_H])
            
            label_train.append(j)
            
    dataset_train = np.asarray(dataset_train, dtype = np.float32)
    
    new_input_data = []
    for x in range(len(dataset_train)):
        new_input_data.append(dataset_train[x].transpose())
    
    new_input_data = np.asarray(new_input_data, dtype = np.float32)


    ilepath = 'dataset/'+ str(i) + '/' + onlyfiles[1]
    aux_data = pd.read_csv(filepath, delimiter = '\t')

    ls_classes = np.unique(aux_data['class'])
    
    print(filepath)
    
    for j in range(1,len(ls_classes)):
        channel1 = aux_data[aux_data['class'] == ls_classes[j]]['channel1']
        channel2 = aux_data[aux_data['class'] == ls_classes[j]]['channel2']
        channel3 = aux_data[aux_data['class'] == ls_classes[j]]['channel3']
        channel4 = aux_data[aux_data['class'] == ls_classes[j]]['channel4']
        channel5 = aux_data[aux_data['class'] == ls_classes[j]]['channel5']
        channel6 = aux_data[aux_data['class'] == ls_classes[j]]['channel6']
        channel7 = aux_data[aux_data['class'] == ls_classes[j]]['channel7']
        channel8 = aux_data[aux_data['class'] == ls_classes[j]]['channel8']
        
        for k in range(len(channel1) - wnd):
            ch_A = channel1[k : k + wnd]
            ch_B = channel2[k : k + wnd]
            ch_C = channel3[k : k + wnd]
            ch_D = channel4[k : k + wnd]
            ch_E = channel5[k : k + wnd]
            ch_F = channel6[k : k + wnd]
            ch_G = channel7[k : k + wnd]
            ch_H = channel8[k : k + wnd]
            
            dataset_test.append([ch_A, ch_B, ch_C, ch_D,\
                            ch_E, ch_F, ch_G, ch_H])
            
            label_test.append(j)
                
    dataset_test = np.asarray(dataset_test, dtype = np.float32)
    
    new_input_data2 = []
    for x in range(len(dataset_test)):
        new_input_data2.append(dataset_test[x].transpose())
        
    new_input_data2 = np.asarray(new_input_data2, dtype = np.float32)





hf = h5py.File('data.h5', 'w')

hf.create_dataset('input_train', data = new_input_data)
hf.create_dataset('output_train', data = label_train)

hf.create_dataset('input_test', data = new_input_data2)
hf.create_dataset('output_test', data = label_test)

hf.close()