##############################################################################
#------------------------- Declare Code Variables ---------------------------#
##############################################################################
n_sensors = 8
n_samples = 500

patiency = 10
verbs = 1

Input_Shape = (n_samples, n_sensors)

##############################################################################
#------------------------------- Load Dataset -------------------------------#
##############################################################################
import numpy as np
import h5py
hf = h5py.File('dataset-treino/data.h5', 'r')

X_train = hf.get('input_train')
X_train = np.array(X_train)
y_train = hf.get('output_train')
y_train = np.array(y_train)

X_test = hf.get('input_test')
X_test = np.array(X_test)
y_test = hf.get('output_test')
y_test = np.array(y_test)

hf.close()

for i in range(len(y_train)):
    y_train[i] = y_train[i] - 1

for i in range(len(y_test)):
    y_test[i] = y_test[i] - 1


##############################################################################
#--------------------------- Scale The dataset ------------------------------#
##############################################################################
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scalers_1 = {}
for i in range(X_train.shape[2]):
    scalers_1[i] = StandardScaler()
    X_train[:, :, i] = scalers_1[i].fit_transform(X_train[:, :, i]) 
for i in range(X_test.shape[2]):
    X_test[:, :, i] = scalers_1[i].fit_transform(X_test[:, :, i])
    
scalers_2 = {}
for i in range(X_train.shape[2]):
    scalers_2[i] = MinMaxScaler(feature_range=(0, 1))
    X_train[:, :, i] = scalers_2[i].fit_transform(X_train[:, :, i])

for i in range(X_test.shape[2]):
    scalers_2[i] = MinMaxScaler(feature_range=(0, 1))
    X_test[:, :, i] = scalers_2[i].fit_transform(X_test[:, :, i])
    
    
##############################################################################
#-------------------- Encode Categorical Data to Keras ----------------------#
##############################################################################
from keras.utils import np_utils
y_hot_train = np_utils.to_categorical(y_train, len(np.unique(y_train)))
y_hot_test = np_utils.to_categorical(y_test, len(np.unique(y_test)))


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
output = Dense(len(np.unique(y_train)),activation='softmax')(hidden2)

model = Model(inputs = visible, outputs = output)

print(model.summary())

##############################################################################
#---------------------- Convolutional Model Parameters ----------------------#
##############################################################################
callbacks_list = [keras.callbacks.EarlyStopping(monitor = 'acc',
                                                patience = patiency,
                                                verbose = verbs)]

model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam', 
              metrics = ['accuracy'])

# ANN - Parameters
BATCH_SIZE = 1000
EPOCHS = 150

##############################################################################
#-------------------------------- ANN Training ------------------------------#
##############################################################################
history = model.fit(X_train, y_hot_train,
                    batch_size = BATCH_SIZE,
                    epochs = EPOCHS,
                    callbacks = callbacks_list,
                    validation_data = (X_test, y_hot_test),
                    verbose = verbs)

model.save_weights("model.h5")