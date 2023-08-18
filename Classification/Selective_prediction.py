import tensorflow as tf
from tensorflow import keras 
import numpy as np
import os 

def data_load(): # DATA Upload 
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

    X_DATASET = np.concatenate((NORMAL, APNEA), axis = 0)
    Y_DATASET = np.array([0] * NORMAL.shape[0] + [1] * APNEA.shape[0])
    
    test_NORMAL = np.load(path + list_[-1] + "/NORMAL.npy")
    test_APNEA = np.load(path + list_[-1] + "/APNEA.npy")
    test_X_DATASET = np.concatenate((test_NORMAL, test_APNEA), axis = 0)
    test_Y_DATASET = np.array([0] * test_NORMAL.shape[0] + [1] * test_APNEA.shape[0])
    
    return X_DATASET, Y_DATASET, test_X_DATASET, test_Y_DATASET

def selective_loss(y_true, y_pred): # Selective prediction loss 
    lamda = 2.0e2
    c = 0.98 # Coverage, hyper parameter
    gt = tf.keras.backend.repeat_elements(y_pred[:,-1:], 1, axis=1) * y_true[:,:] # activated (survived) label
    pred = y_pred[:,:-1]
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt, logits=pred)) + lamda * tf.keras.backend.maximum(-tf.reduce_mean(y_pred[:,-1]) + c, 0) ** 2
    return loss

class SelAnet:
    
    model = None
    def __init__(self,
                 lr = 0.001,
                 beta_1 = 0.9,
                 beta_2 = 0.999,
                 epsilon = 1e-08,
                 loss_weight = [0.4 , 0.6],
                 loss = {'Auxil_output': tf.keras.losses.BinaryCrossentropy(), 
                         'Selective_prediction': selective_loss},
                 filters = 8,
                 ):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.loss_weight = loss_weight
        self.loss = loss
        self.filters = filters
    
    def Model(self):
        input_ = tf.keras.Input(shape = (150, 8), name = 'Input')
        reshpae_input_ = keras.layers.Reshape((150,8,1))(input_)
        
        # Prediction
        conv1 = tf.keras.layers.TimeDistributed(keras.layers.Conv1D(filters=self.filters, kernel_size=2, activation='relu', 
                                                                     name = "Convolution_1", padding='same'))(reshpae_input_)
        pool1 = tf.keras.layers.TimeDistributed(keras.layers.MaxPooling1D(pool_size=2), name = 'Pooling_1')(conv1)
        flatten = keras.layers.TimeDistributed(keras.layers.Flatten(), name = 'Flatten_layer')(pool1) 
        lstm1 = tf.keras.layers.LSTM(16, name = 'Lstm_layer')(flatten)
        pre_output = keras.layers.Dense(1, name = 'prediction')(lstm1)
        Auxil_output = keras.layers.Activation('sigmoid', name = 'Auxil_output')(pre_output)
        
        # Selection
        Selec_layer = keras.layers.Dense(10, name = "Selection_1")(lstm1)
        Selec_relu = keras.layers.Activation('relu', name = 'Selection_relu')(Selec_layer)
        Selec_BN = keras.layers.BatchNormalization(name = 'Selection_BN')(Selec_relu)
        Selec_layer_2 = keras.layers.Dense(1, name = "Selection_2")(Selec_BN) 
        Selec_output = keras.layers.Activation('sigmoid', name = 'Selection_output')(Selec_layer_2)

        Selective_prediction = keras.layers.Concatenate(axis = 1, name = 'Selective_prediction')([pre_output, Selec_output]) 
        model = keras.Model(inputs = input_, outputs = [Auxil_output, Selective_prediction])
        
        optimizer = keras.optimizers.Adam(lr = self.lr, beta_1 = self.beta_1, beta_2 = self.beta_2, epsilon = self.epsilon, decay = 0.0)
        model.compile(optimizer = optimizer, loss = self.loss, metrics = ['accuracy'], loss_weights= self.loss_weight)
        self.model = model
        return model


def Train(batch_size = 64, epochs = 300):
    Train_x, Train_y, Test_x, Test_y = data_load()    
    Train_y = Train_y.astype('float32')
    Test_y = Test_y.astype('float32') 
    
    model = SelAnet().Model()
    callbacks = []

    model_file = 'file_name'
    callbacks.append(keras.callbacks.ModelCheckpoint(model_file, monitor = 'loss', save_best_only = True, save_weight_only = False))
      
    model.fit(Train_x, Train_y, batch_size = batch_size, epochs = epochs, callbacks = callbacks, verbose = 1, 
              validation_data = (Test_x, Test_y)) 
