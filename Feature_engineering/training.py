# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 22:08:38 2024

@author: user
"""
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tensorflow import keras
import datetime
from tcn import TCN
from TCNAE import TCNAE


# data load
def data_shuffle(data):
    _, X_test = train_test_split(data, test_size=0.1, shuffle=True)
    return X_test

def load_data(mode, data_type):
    if data_type == 'sao2':
        base_path = 'D:/연구_최종/SAO2/'
    elif data_type == 'resp':
        base_path = 'D:/연구_최종_결과분석/EDR/'
    else:
        raise ValueError("Invalid data_type. Choose 'sao2' or 'resp'.")

    path = os.path.join(base_path, mode)
    file_list = os.listdir(path)

    dataset = np.load(os.path.join(path, file_list[0], f"{file_list[0]}.npy"))
    dataset = data_shuffle(dataset)

    for i, file in enumerate(file_list[1:600], 1):
        print(f"{file}, {i}")
        file_path = os.path.join(path, file, f"{file}.npy")
        data = np.load(file_path)
        
        if data.size > 0:
            data = data_shuffle(data)
            dataset = np.concatenate((dataset, data), axis=0)

    dataset = dataset[~np.isnan(dataset)]
    return dataset.reshape(-1, 6000, 1)

# dataset 
def load_and_preprocess_data(mode):
    sao2_data = load_data('normal', 'sao2')
    resp_data = load_data('normal', 'resp')
    data_set = np.concatenate((sao2_data, resp_data), axis=-1)
    return data_set.reshape(-1, 6000, 2)


# for training
def train(mode):
    # Load and preprocess data
    train_set = load_and_preprocess_data(mode)
    train_set, test_set = train_test_split(train_set, test_size=0.2, shuffle=True, random_state=10)
    print(f"Training set shape: {train_set.shape}")

    # Load pre-trained model
    M_PATH = 'autoencoder_model.h5'
    model = keras.models.load_model(M_PATH, custom_objects={'TCN': TCN})

    # Set up callbacks
    d = datetime.datetime.now()
    model_file = f'new_autoencoder_model.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_file, monitor='loss', save_best_only=True, save_weights_only=False),
        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    ]

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    # Train model
    model.fit(train_set, train_set, batch_size=32, epochs=30, callbacks=callbacks, validation_data=(test_set, test_set), verbose=1)

# Uncomment to train the model
# train("APNEA")

# Create and display model summary
model = TCNAE().build_model()
model.summary()