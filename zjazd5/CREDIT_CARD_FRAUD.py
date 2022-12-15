""""
Authors: Zuzanna Ciborowska s20682 & Joanna Walkiewicz s20161
Python project to cheking if credit card transaction was fraud using neural network with 1 and 3 layers
System requirements:
- Python 3.10
- Tensorflow
- Numpy
- Keras
- Sckit-learn
- Pandas
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import Normalizer
from keras.layers import Activation, Dense, Dropout, BatchNormalization, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

"""
Read the file
"""
df = pd.read_csv("card_transdata_smaller.csv")


"""
Split data for test and train data
"""
X = df.drop('fraud', axis =1).values
y = df.fraud.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
nl = Normalizer()
nl.fit(X_train)
X_train = nl.transform(X_train)

"""
Define neural network with 3 hidden layers
"""
def nn3():
    inputs = Input(name='inputs', shape=[X_train.shape[1]])
    layer = Dense(128, name='FC1')(inputs)
    layer = BatchNormalization(name='BC1')(layer)
    layer = Activation('relu', name='Activation1')(layer)
    layer = Dropout(0.3, name='Dropout1')(layer)
    layer = Dense(128, name='FC2')(layer)
    layer = BatchNormalization(name='BC2')(layer)
    layer = Activation('relu', name='Activation2')(layer)
    layer = Dropout(0.3, name='Dropout2')(layer)
    layer = Dense(128, name='FC3')(layer)
    layer = BatchNormalization(name='BC3')(layer)
    layer = Dropout(0.3, name='Dropout3')(layer)
    layer = Dense(1, name='OutLayer')(layer)
    layer = Activation('sigmoid', name='sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


"""
Create and compile the model
"""
model = nn3()
#model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

"""
Define callbacks
"""
reduce_lr = ReduceLROnPlateau()
early_stopping = EarlyStopping(patience=20, min_delta=0.0001)

"""
Fit the model
"""
model.fit(x=X_train, y=y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping], verbose=0)

"""
Create metrics
"""
x_lst = [X_train, X_test]
y_lst = [y_train, y_test]
for i,(x,y) in enumerate(zip(x_lst, y_lst)):
    y_pred = model.predict(x)
    y_pred = np.around(y_pred)
    y_pred = np.asarray(y_pred)
    if i == 0:
        print('Training set:')
        print('\tAccuracy:{:0.3f}\n\tClassification Report\n{}'.format(accuracy_score(y, y_pred),
                                                                  classification_report(y, y_pred)))
    else:
        print('Test set:')
        print('\tAccuracy:{:0.3f}\n\tClassification Report\n{}'.format(accuracy_score(y, y_pred),
                                                                  classification_report(y, y_pred)))

print("------------------------------------------------------")

"""
Define neural network with 1 hidden layer
"""
def nn1():
    inputs = Input(name='inputs', shape=[X_train.shape[1]])
    layer = Dense(128, name='FC1')(inputs)
    layer = BatchNormalization(name='BC1')(layer)
    layer = Activation('relu', name='Activation1')(layer)
    layer = Dropout(0.3, name='Dropout1')(layer)
    layer = Dense(1, name='OutLayer')(layer)
    layer = Activation('sigmoid', name='sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


"""
Create and compile the model
"""
model = nn1()
#model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

"""
Define callbacks
"""
reduce_lr = ReduceLROnPlateau()
early_stopping = EarlyStopping(patience=20, min_delta=0.0001)

"""
Fit the model
"""
model.fit(x=X_train, y=y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping], verbose=0)

"""
Create metrics
"""
x_lst = [X_train, X_test]
y_lst = [y_train, y_test]
for i,(x,y) in enumerate(zip(x_lst, y_lst)):
    y_pred = model.predict(x)
    y_pred = np.around(y_pred)
    y_pred = np.asarray(y_pred)
    if i == 0:
        print('Training set:')
        print('\tAccuracy:{:0.3f}\n\tClassification Report\n{}'.format(accuracy_score(y, y_pred),
                                                                  classification_report(y, y_pred)))
    else:
        print('Test set:')
        print('\tAccuracy:{:0.3f}\n\tClassification Report\n{}'.format(accuracy_score(y, y_pred),
                                                                  classification_report(y, y_pred)))
