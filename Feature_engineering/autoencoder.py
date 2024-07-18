import tensorflow as tf
from tensorflow import keras
from tcn import TCN

class TCNAE:
    """
    Temporal Convolutional Autoencoder (TCN-AE) model.
    """
    
    def __init__(self,
                 ts_dimension=2,
                 dilations=(1, 2, 4, 8, 16),
                 nb_filters=40,
                 kernel_size=10,
                 nb_stacks=1,
                 padding='same',
                 dropout_rate=0.00,
                 filters_conv1d=8,
                 activation_conv1d='relu',
                 latent_sample_rate=40,
                 pooler=keras.layers.AveragePooling1D,
                 lr=0.001,
                 conv_kernel_init='glorot_normal',
                 loss='mse',
                 use_early_stopping=False,
                 error_window_length=128,
                 verbose=1):
        
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
        
        self.model = self.build_model(verbose)
    
    def build_model(self, verbose=1):
        """Builds the TCN-AE model."""
        
        tf.keras.backend.clear_session()
        
        inputs = keras.Input(shape=(6000, self.ts_dimension))
        
        # Encoder
        x = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, 
                nb_stacks=self.nb_stacks, dilations=self.dilations, 
                padding=self.padding, use_skip_connections=True, 
                dropout_rate=self.dropout_rate, return_sequences=True,
                kernel_initializer=self.conv_kernel_init, name='tcn-enc')(inputs)
        
        x = keras.layers.Conv1D(filters=self.filters_conv1d, kernel_size=1, 
                                activation=self.activation_conv1d, padding=self.padding)(x)
        
        x = self.pooler(pool_size=self.latent_sample_rate)(x)
        encoded = keras.layers.Activation("relu")(x)
        
        # Decoder
        x = keras.layers.UpSampling1D(size=self.latent_sample_rate)(encoded)
        
        x = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, 
                nb_stacks=self.nb_stacks, dilations=self.dilations, 
                padding=self.padding, use_skip_connections=True, 
                dropout_rate=self.dropout_rate, return_sequences=True,
                kernel_initializer=self.conv_kernel_init, name='tcn-dec')(x)
        
        outputs = keras.layers.Dense(self.ts_dimension, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        
        if verbose > 1:
            model.summary()
        
        return model