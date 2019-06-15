# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:54:51 2019

@author: eduar
"""
import sys
import numpy as np
from sklearn import metrics
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE

def plot_confusion_matrix(val_real, val_pred, title, labels, norm, language):
    if language == 'English':
        y_label = 'True Label'
        x_label = 'Predicted Label'
    elif language == 'Portuguese':
        y_label = 'Valor real'
        x_label = 'Valor previsto'
        
    matrix = metrics.confusion_matrix(val_real, val_pred)
    plt.figure(figsize=(8, 8))
    
    if norm == False:    
        sns.heatmap(matrix,
                    cmap='coolwarm',
                    linecolor='white',
                    linewidths=1,
                    xticklabels=labels,
                    yticklabels=labels,
                    annot=True,
                    fmt='d',
                    square=False)
        
    elif norm == True:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        sns.heatmap(matrix,
                    cmap='coolwarm',
                    linecolor='white',
                    linewidths=1,
                    xticklabels=labels,
                    yticklabels=labels,
                    annot=True,
                    fmt='.2f',
                    square=False)

    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    
    
    
def tSNE_plot(SNE_input, SNE_output, n_data, classes, labels, colors,\
          mode, title, marker):
    x_SNE = []
    y_SNE = []
        
    for j in range(classes):
        out = 0
        counter = 0
        i = 0
        while out == 0:
            if SNE_output[i] == j:
                x_SNE.append(SNE_input[i,:])
                y_SNE.append(SNE_output[i])
                counter = counter + 1
            if counter == n_data:
                out = 1
            i = i + 1
    x_SNE = np.asarray(x_SNE)
    y_SNE = np.asarray(y_SNE)
    if mode == '2D':
        dimensions = 2
    elif mode == '3D':
        dimensions = 3
        
    tsne = TSNE(n_components = dimensions)
    X = tsne.fit_transform(x_SNE)
    
    if mode == '2D':
        plt.figure(figsize = (8, 8))
        for i in range(0,len(labels)):
            plt.scatter(X[y_SNE == i, 0], X[y_SNE == i, 1],\
                        label = labels[i], c = colors[i],\
                        marker = marker[i])
        plt.legend(loc = 'best')
        plt.title(title)
        
    elif mode == '3D':    
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(0,len(labels)):
            ax.scatter(X[y_SNE == i, 0], X[y_SNE == i, 1], X[y_SNE == i, 2],\
                        label = labels[i], c = colors[i], \
                        marker = marker[i])
        ax.legend(loc = 'best')
        plt.title(title)

##############################################################################
#------------------------- Declare Code Variables ---------------------------#
##############################################################################
n_sensors = 8
n_samples = 500

Input_Shape = (n_samples, n_sensors)

##############################################################################
#------------------------------- Load Dataset -------------------------------#
##############################################################################
import numpy as np
import h5py
hf = h5py.File('dataset-treino/data.h5', 'r')

X_test = hf.get('input_test')
X_test = np.array(X_test)
y_test = hf.get('output_test')
y_test = np.array(y_test)

hf.close()

for i in range(len(y_test)):
    y_test[i] = y_test[i] - 1

##############################################################################
#----------------------- Build the Convolutional Model ----------------------#
##############################################################################
import keras
from keras.models import Input, Model
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D

visible = Input(shape = Input_Shape)

conv1 = Conv1D(32, 3, activation='relu')(visible)
conv2 = Conv1D(32, 3, activation='relu')(conv1)
pool1 = MaxPooling1D(pool_size = 3)(conv2)

conv3 = Conv1D(32, 3, activation='relu')(pool1)
conv4 = Conv1D(32, 3, activation='relu')(conv3)
pool2 = MaxPooling1D(pool_size = 3)(conv4)

conv5 = Conv1D(32, 3, activation='relu')(pool2)
conv6 = Conv1D(32, 3, activation='relu')(conv5)
pool3 = MaxPooling1D(pool_size = 3)(conv6)

flat = Flatten()(pool3)

hidden1 = Dense(256, activation='relu')(flat)
hidden2 = Dense(128, activation='relu')(hidden1)
output = Dense(len(np.unique(y_test)),activation='softmax')(hidden2)

model = Model(inputs = visible, outputs = output)

print(model.summary())

##############################################################################
#------------------------------- Load Weights -------------------------------#
##############################################################################
from keras.models import load_model
model.load_weights('rede-treinada/model.h5')

##############################################################################
#--------------------------- Scale The dataset ------------------------------#
##############################################################################
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scalers_1 = {}
for i in range(X_test.shape[2]):
    scalers_1[i] = StandardScaler()
    X_test[:, :, i] = scalers_1[i].fit_transform(X_test[:, :, i])
    
scalers_2 = {}
for i in range(X_test.shape[2]):
    scalers_2[i] = MinMaxScaler(feature_range=(0, 1))
    X_test[:, :, i] = scalers_2[i].fit_transform(X_test[:, :, i])

##############################################################################
#-------------------- Encode Categorical Data to Keras ----------------------#
##############################################################################
# Evaluate the train result
y_pred_test = model.predict(X_test)
y_pred = np.argmax(y_pred_test, axis=1)

labels = 'A', 'B', 'C', 'D', 'E', 'F'

plot_confusion_matrix(y_test, y_pred, 'Matriz de Confusão', labels, False,\
                      'Portuguese')

##############################################################################
#---------------------------- Get parameters after CNN ----------------------#
##############################################################################
# Load Convolutional Model
extract_model = Model(inputs = model.input, outputs =\
                      model.get_layer('flatten_1').output)

# Take the convolutional filter output
test_filter_output = extract_model.predict(X_test)
colors = 'r', 'g', 'b', 'y', 'k', 'orange'
marker = '.','.','.','.','.','.'

tSNE_plot(test_filter_output, y_test, 200, 6, labels, colors,\
              '2D', 't-SNE', marker)