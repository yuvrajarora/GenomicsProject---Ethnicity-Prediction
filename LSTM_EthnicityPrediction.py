import os
import pandas as pd
import glob
import numpy as np
import sys
from time import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from keras.utils import np_utils

from keras import backend as K
K.set_image_dim_ordering('tf')
import matplotlib.pyplot as plt
import itertools

import numpy as np
import pandas as pd
np.random.seed(10)

from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import RMSprop
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn import manifold
import keras.layers.normalization as bn

from sklearn.metrics import confusion_matrix

df1 = pd.read_csv('y_chr.csv', header=None)
print(df1.head())

label = df1[0]
print(label.head())

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
lbl = le.fit(label)
labelss = lbl.transform(label)
labelDF = pd.DataFrame(labelss)

#labelArr = 
print(labelDF.head())

feature = df1.drop(0, axis=1)
print(feature.head())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x1 = feature.iloc[:,1:]
df_scaled = pd.DataFrame(scaler.fit_transform(x1), columns=x1.columns)
df_scaled.head()

y = labelss
x = df_scaled.values

features = x
labels = y

def prepare_test_train_valid():
	# Train-test split 
	train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.15, random_state=100)
	train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.176471, random_state=100)

	return train_x, test_x, train_y, test_y, valid_x, valid_y

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

labels = one_hot_encode(labels)

# Extract feature
train_x, test_x, train_y, test_y, valid_x, valid_y = prepare_test_train_valid()

print('X_train shape:', train_x.shape)
print('Y_train shape:', train_y.shape)

num_classes = 5
data_dim = 15091
timesteps = 1

train_x = np.reshape(train_x,(train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x,(test_x.shape[0], 1, test_x.shape[1]))
valid_x = np.reshape(valid_x,(valid_x.shape[0], 1, valid_x.shape[1]))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def build_LSTM(): #OK
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim))) 
	
    model.add(LSTM(24, return_sequences=True))

    #model.add(Dropout(0.2))
    model.add(LSTM(16, return_sequences=True)) 
    model.add(Dropout(0.2))
    
    # apply softmax to output
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return model

def model_train_evaluate(model, number_epoch):   
    sgd = RMSprop(lr=0.001, rho=0.01, epsilon=None, decay=0.0)
    optm = Adam(lr=1e-3)
    # a stopping function should the validation loss stop improving
    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

    #if model in ['RNN']: 
    rnn_model = build_LSTM() #OK
    rnn_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optm)
    #tensorboardRNN = TensorBoard(log_dir="RNN_logs/{}".format(time()))
    h = rnn_model.fit(train_x, train_y, validation_data=(valid_x, valid_y), batch_size=30, epochs=int(number_epoch))
    print(rnn_model.summary())
        
    y_prob = rnn_model.predict(test_x) 
    y_pred = y_prob.argmax(axis=-1)
    y_true = np.argmax(test_y, 1)

    roc = roc_auc_score(test_y, y_prob)
    print ("ROC:",  round(roc,3))

    # evaluate the model
    score, accuracy = rnn_model.evaluate(test_x, test_y, batch_size=35)
    print("\nAccuracy = {:.2f}".format(accuracy))

    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    
    class_names = ["BEB", "PJL", "ESN", "IBS", "PUR"]
    #Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix: true vs predicted label')  
    # plt.show()
    
    plt.figure(1, figsize=(9,9))
    plt.suptitle('LSTM - CNN - 128 - 256 - 128 Dropout 0.5', fontsize = 15)
    plt.subplot(211)  
    plt.plot(h.history['acc'], linewidth = 1.0)  
    plt.plot(h.history['val_acc'], linewidth = 1.0)  
    plt.title('Accuracy')  
    plt.ylabel('Accuracy')  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'test'], loc='best') 

    plt.subplot(212)  
    plt.plot(h.history['loss'], linewidth = 1.0)  
    plt.plot(h.history['val_loss'], linewidth = 1.0)  
    plt.title('Loss')  
    plt.ylabel('Loss')  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'test'], loc='best')  
    plt.show()

model = build_LSTM()
model_train_evaluate(model, 30)
import gc; gc.collect()
