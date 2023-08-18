import numpy
from tcn import TCN
import tensorflow
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import os 
from sklearn.model_selection import train_test_split


class TCNAE:
    
    model = None
    
    def __init__(self,
                 ts_dimension = 2,
                 dilations = (1, 2, 4, 8, 16),
                 nb_filters = 40,
                 kernel_size = 10,
                 nb_stacks = 1,
                 padding = 'same',
                 dropout_rate = 0.00,
                 filters_conv1d = 8,
                 activation_conv1d = 'relu',  
                 latent_sample_rate = 40,
                 pooler = AveragePooling1D,
                 lr = 0.001,
                 conv_kernel_init = 'glorot_normal',
                 loss = 'mse', 
                 use_early_stopping = False,
                 error_window_length = 128,
                 verbose = 1
                ):

        self.ts_dimension = ts_dimension
        self.dilations = dilations
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.filters_conv1d = filters_conv1d
        self.activation_conv1d = activation_conv1d
        self.latent_sample_rate = latent_sample_rate
        self.pooler = pooler
        self.lr = lr
        self.conv_kernel_init = conv_kernel_init
        self.loss = loss
        self.use_early_stopping = use_early_stopping
        self.error_window_length = error_window_length
        
        self.build_model(verbose = verbose)
        
    
    def build_model(self, verbose = 1):
        
        tensorflow.keras.backend.clear_session()
        sampling_factor = self.latent_sample_rate
        i = Input(batch_shape=(None, 6000, self.ts_dimension))

        tcn_enc = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, nb_stacks=self.nb_stacks, dilations=self.dilations, 
                      padding=self.padding, use_skip_connections=True, dropout_rate=self.dropout_rate, return_sequences=True,
                      kernel_initializer=self.conv_kernel_init, name='tcn-enc')(i)

        enc_flat = Conv1D(filters=self.filters_conv1d, kernel_size=1, activation=self.activation_conv1d, padding=self.padding)(tcn_enc)
        enc_pooled = self.pooler(pool_size=sampling_factor, strides=None, padding='valid', data_format='channels_last')(enc_flat)
        enc_out = Activation("relu")(enc_pooled)
        dec_upsample = UpSampling1D(size=sampling_factor)(enc_out)
        
        dec_reconstructed = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, nb_stacks=self.nb_stacks, dilations=self.dilations, 
                                padding=self.padding, use_skip_connections=True, dropout_rate=self.dropout_rate, return_sequences=True,
                                kernel_initializer=self.conv_kernel_init, name='tcn-dec')(dec_upsample)
        
        o = Dense(self.ts_dimension, activation='linear')(dec_reconstructed)  

        model = Model(inputs=[i], outputs=[o])

        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
        model.compile(loss=self.loss, optimizer=adam, metrics=[self.loss, 'accuracy'])
        self.model = model
        return model 

# =============================================================================
# DATA Load
# =============================================================================

def data_shuffle(data):
    X_train, X_test = train_test_split(data, test_size=0.1, shuffle = True)
    return X_test

def sao2_load_data():  
    path = 'path'
    a_path = path + '/'
    a_list = os.listdir(a_path)
    dataset = numpy.load(a_path + a_list[0] + '/' + a_list[0] + '.npy') 
    dataset = data_shuffle(dataset)
    for i in a_list[1:600]:
        print(i, a_list.index(i))
        a_data = numpy.load(a_path + i + '/' + i + '.npy')
        if a_data.size > 0:
            a_data = data_shuffle(a_data)
            dataset = numpy.concatenate((dataset, a_data), axis = 0)
        else:
            pass 
    dataset = numpy.array(dataset)
    dataset = dataset[numpy.logical_not(numpy.isnan(dataset))]
    dataset = dataset.reshape(-1, 6000, 1)
    return dataset

def resp_load_data():
    path = 'path'
    a_path = path + '/'
    
    a_list = os.listdir(a_path)
    dataset = numpy.load(a_path + a_list[0] + '/' + a_list[0] + '.npy') 
    dataset = data_shuffle(dataset)
    for i in a_list[1:600]:
        print(i, a_list.index(i))
        a_data = numpy.load(a_path + i + '/' + i + '.npy')
        if a_data.size > 0:
            a_data = data_shuffle(a_data)
            dataset = numpy.concatenate((dataset, a_data), axis = 0)
        else:
            pass 
    dataset = numpy.array(dataset)
    dataset = dataset[numpy.logical_not(numpy.isnan(dataset))]
    dataset = dataset.reshape(-1, 6000, 1)
    return dataset

def train(batch_size, epochs):
    sao2 = sao2_load_data()
    resp = resp_load_data()
    data_set = numpy.concatenate((sao2, resp), axis = -1)
    train_set = data_set.reshape(-1,6000,2)
    train_set, test_set = train_test_split(train_set, test_size = 0.2, shuffle = True, random_state = 10)  

    model = TCNAE().build_model()
    callbacks = []
    
    model_file = "path/TCNAE.h5"
    callbacks.append(tensorflow.keras.callbacks.ModelCheckpoint(model_file, monitor = 'loss', 
                                                                save_best_only = True, save_weight_only = False))
    callbacks.append(tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, 
                                                                  patience=3, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0))
    
    model.fit(train_set, train_set, batch_size = batch_size, epochs = epochs, 
              callbacks = callbacks, validation_data = (test_set,test_set), verbose = 1)
    
