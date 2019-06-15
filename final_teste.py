import sys
import numpy as np

##############################################################################
#------------------------- Declare Code Variables ---------------------------#
##############################################################################
n_sensors = 8
n_samples = 500

Input_Shape = (n_samples, n_sensors)

##############################################################################
#------------------------------- Load Dataset -------------------------------#
##############################################################################
import pandas as pd
filename = sys.argv[1]
X_test = pd.read_csv(filename)
X_test = np.asarray(X_test).reshape(1,n_samples,n_sensors)

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
output = Dense(6,activation='softmax')(hidden2)

model = Model(inputs = visible, outputs = output)

print(model.summary())

##############################################################################
#------------------------------- Load Weights -------------------------------#
##############################################################################
model.load_weights('rede-treinada/model.h5')

##############################################################################
#------------------------------ Predict Class -------------------------------#
##############################################################################
# Evaluate the train result
y_pred_test = model.predict(X_test)
y_pred = np.argmax(y_pred_test, axis=1)

y_pred = y_pred + 1

##############################################################################
#-------------------------------- Plot Image --------------------------------#
##############################################################################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



if y_pred == 1:
    img=mpimg.imread('images/1_nada.jpg')
    imgplot = plt.imshow(img)
    plt.title('Nada')
    plt.show()
    
elif y_pred == 2:
    img=mpimg.imread('images/2_soquinho.jpg')
    imgplot = plt.imshow(img)
    plt.title('Soquinho')
    plt.show()
    
elif y_pred == 3:
    img=mpimg.imread('images/3_munheca.jpg')
    imgplot = plt.imshow(img)
    plt.title('Quebrando Munheca')
    plt.show()
elif y_pred == 4:
    img=mpimg.imread('images/4_munheca_pra_cima.jpg')
    imgplot = plt.imshow(img)
    plt.title('Olhando a unha')
    plt.show()
elif y_pred == 5:
    img=mpimg.imread('images/5_tchauzinho.jpg')
    imgplot = plt.imshow(img)
    plt.title('Tchauzinho')
    plt.show()
elif y_pred == 6:
    img=mpimg.imread('images/6_dedim_de_ladim.jpg')
    imgplot = plt.imshow(img)
    plt.title('Dedim de Ladim')
    plt.show()
        
