import tensorflow as tf
from tensorflow import keras

def Total_model():
    # Input layer
    input_layer = tf.keras.Input(shape=(150, 8), name="Input")
    reshaped_input = keras.layers.Reshape((150, 8, 1))(input_layer)

    # Prediction branch
    conv1 = tf.keras.layers.TimeDistributed(
        keras.layers.Conv1D(filters=8, kernel_size=2, activation='relu', padding='same')
    )(reshaped_input)
    pool1 = tf.keras.layers.TimeDistributed(keras.layers.MaxPooling1D(pool_size=2))(conv1)
    flatten_2 = keras.layers.TimeDistributed(keras.layers.Flatten())(pool1)
    lstm1 = tf.keras.layers.LSTM(32)(flatten_2)  # 1D CNN-LSTM
    c_layer = keras.layers.Dense(1, name='prediction')(lstm1)
    auxil_output = keras.layers.Activation('sigmoid', name='Auxil_output')(c_layer)

    # Selection branch
    s_layer_1 = keras.layers.Dense(10, name="S_layer_1")(lstm1)
    s_relu = keras.layers.Activation('relu', name='S_relu')(s_layer_1)
    s_normal = keras.layers.BatchNormalization()(s_relu)
    s_layer_4 = keras.layers.Dense(1, name="S_layer_4")(s_normal)
    selection_output = keras.layers.Activation('sigmoid', name='Selection_output')(s_layer_4)

    # Combine prediction and selection outputs
    selective_prediction = keras.layers.Concatenate(axis=1, name='Selective_prediction')(
        [c_layer, selection_output]
    )

    # Create and return the model
    model = keras.Model(
        inputs=input_layer,
        outputs=[auxil_output, selective_prediction]
    )
    model.summary()
    return model