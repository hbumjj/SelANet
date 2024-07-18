# model structure

# import module
import tensorflow as tf
from tensorflow import keras 
import os 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# data load
def data_load():
    path = 'PATH'
    list_ = os.listdir(path)
    
    # Initialize with first file
    NORMAL = np.load(os.path.join(path, list_[1], "NORMAL.npy"))
    APNEA = np.load(os.path.join(path, list_[1], "APNEA.npy"))
    
    # Load and concatenate remaining files
    for file in list_[2:]:
        file_path = os.path.join(path, file)
        NORMAL = np.concatenate((NORMAL, np.load(os.path.join(file_path, "NORMAL.npy"))))
        APNEA = np.concatenate((APNEA, np.load(os.path.join(file_path, "APNEA.npy"))))
    
    # Combine datasets
    X_DATASET = np.concatenate((NORMAL, APNEA))
    Y_DATASET = np.concatenate((np.zeros(NORMAL.shape[0]), np.ones(APNEA.shape[0])))
    
    print(f"X_DATASET shape: {X_DATASET.shape}, Y_DATASET shape: {Y_DATASET.shape}")
    
    return X_DATASET, Y_DATASET
    
# selective prediction loss
def selective_loss(y_true, y_pred):
    lamda = 2.0e2
    c = 0.98  # coverage
 
    # calculate activated label
    gt = tf.keras.backend.repeat_elements(y_pred[:,-1:], 2, axis=1) * y_true[:,:]
    
    # prediction
    pred = y_pred[:,:-1]
    
    # Calculating cross-entropy loss
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=pred))
    
    # Calculating coverage penalties
    coverage_penalty = lamda * tf.keras.backend.maximum(-tf.reduce_mean(y_pred[:,-1]) + c, 0) ** 2
    
    # Calculating the final loss
    loss = ce_loss + coverage_penalty
    
    return loss

# for training
def Train():
    # Load and preprocess data
    Train_x, Train_y = data_load()
    Train_y = keras.utils.to_categorical(Train_y.astype('float32'), 2)

    # Define loss functions
    CC = tf.keras.losses.BinaryCrossentropy()
    losses = {
        'Auxil_output': CC,
        'Selective_prediction': selective_loss
    }

    # Load pre-trained model # continue_training
    model_path = "model.h5"
    custom_objects = {'cce': CC, 'selective_loss': selective_loss}
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    model.summary()

    # Set up callbacks
    model_file = "new_model.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_file,
            monitor='loss',
            save_best_only=True,
            save_weights_only=False
        )
    ]

    # Configure optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=0.0005,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        decay=0.0
    )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=['accuracy'],
        loss_weights=[1, 0]
    )

    # Train model
    model.fit(
        Train_x,
        Train_y,
        batch_size=32,
        epochs=300,
        callbacks=callbacks,
        verbose=1
    )