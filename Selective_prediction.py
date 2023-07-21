import tensorflow as tf
from tensorflow import keras 
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt 
from sklearn import metrics
import gc 
import os 
from sklearn.metrics import roc_curve, f1_score, roc_auc_score, auc
from tcn import TCN
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from scipy import stats

def data_load(): # Data upload 
    path = 'path'
    list_ = os.listdir(path)
    NORMAL = np.load(path + list_[0] + "/NORMAL.npy")
    APNEA = np.load(path + list_[0] + "/APNEA.npy")
    for file in list_[1:-1]:
        normal = np.load(path + file + "/NORMAL.npy")
        apnea = np.load(path + file + "/APNEA.npy")
        NORMAL = np.concatenate((NORMAL, normal), axis = 0)
        APNEA = np.concatenate((APNEA, apnea), axis = 0)
    NORMAL = np.array(NORMAL)
    APNEA = np.array(APNEA)

    print(NORMAL.shape)
    print(APNEA.shape)
    X_DATASET = np.concatenate((NORMAL, APNEA), axis = 0)
    Y_DATASET = np.array([0] * NORMAL.shape[0] + [1] * APNEA.shape[0])
    print(X_DATASET.shape, Y_DATASET.shape)
    
    test_NORMAL = np.load(path + list_[-1] + "/NORMAL.npy")
    test_APNEA = np.load(path + list_[-1] + "/APNEA.npy")
    test_X_DATASET = np.concatenate((test_NORMAL, test_APNEA), axis = 0)
    test_Y_DATASET = np.array([0] * test_NORMAL.shape[0] + [1] * test_APNEA.shape[0])
    print(test_X_DATASET.shape, test_Y_DATASET.shape)
    
    return X_DATASET, Y_DATASET, test_X_DATASET, test_Y_DATASET

def selective_loss(y_true, y_pred): # Selective prediction loss 
    lamda = 2.0e2
    c = 0.98 # coverage
 
    gt = tf.keras.backend.repeat_elements(y_pred[:,-1:], 2, axis=1) * y_true[:,:] # activated (survived) label
    pred = y_pred[:,:-1]
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=pred)) + lamda * tf.keras.backend.maximum(-tf.reduce_mean(y_pred[:,-1]) + c, 0) ** 2
    return loss

def Selective_prediction():
    Input = tf.keras.Input(shape = (150, 8), name = "Input")  
    RESHAPE_1 = keras.layers.Reshape((150,8,1))(Input)
    # prediction
    CONV1 = tf.keras.layers.TimeDistributed(keras.layers.Conv1D(filters=8, kernel_size=2, activation='relu', padding='same'))(RESHAPE_1)
    pool1 = tf.keras.layers.TimeDistributed(keras.layers.MaxPooling1D(pool_size=2))(CONV1)
    Flatten_2 = keras.layers.TimeDistributed(keras.layers.Flatten())(pool1) 
    lstm1 = tf.keras.layers.LSTM(16)(Flatten_2)  # 1D CNN-LSTM  # 16-->32 
    C_LAYER = keras.layers.Dense(1, name = 'prediction')(lstm1)
    Auxil_output = keras.layers.Activation('sigmoid', name = 'Auxil_output')(C_LAYER) # Auxiliary Prediction
    
    S_layer_1 = keras.layers.Dense(10, name = "S_layer_1")(lstm1) 
    S_relu = keras.layers.Activation('relu', name = 'S_relu')(S_layer_1)
    S_normal = keras.layers.BatchNormalization()(S_relu)
    S_layer_4 = keras.layers.Dense(1, name = "S_layer_4")(S_normal) ## 수정필요
    Selection_output = keras.layers.Activation('sigmoid', name = 'Selection_output')(S_layer_4)
    
    Selective_prediction = keras.layers.Concatenate(axis = 1, name = 'Selective_prediction')([C_LAYER, Selection_output]) # Auxil_output --> C_layer_2
    
    model = keras.Model(inputs = Input, outputs = [Auxil_output, Selective_prediction])
    model.summary()  
    return model

def Train():
    Train_x, Train_y, Test_x, Test_y = data_load()
    Train_y = Train_y.astype('float32')
    Test_y = Test_y.astype('float32')    
    Train_y = keras.utils.to_categorical(Train_y, 2)
    Test_y = keras.utils.to_categorical(Test_y, 2)
     
    model = Selective_prediction()
    CC = tf.keras.losses.BinaryCrossentropy()
     
    losses = {'Auxil_output': CC, 'Selective_prediction': selective_loss}
    callbacks = []

    model_file = 'file_name'
    callbacks.append(keras.callbacks.ModelCheckpoint(model_file, monitor = 'loss', save_best_only = True, save_weight_only = False))
      
    optimizer1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
     
    model.compile(optimizer = optimizer1, loss = losses, metrics = ['accuracy'], loss_weights = [0.4, 0.6])
    model.fit(Train_x, Train_y, batch_size = 64, epochs = 300, callbacks = callbacks, verbose = 1, 
              validation_data = (Test_x, Test_y)) 

Train()    