import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import os
from tcn import TCN

# autoencoder load for test
def load_autoencoder():
    model_path = 'autoencoder_model.h5'
    model = keras.models.load_model(model_path, custom_objects={'TCN': TCN})
    model.layers[4]._name = "encoder"
    encoder_output = model.get_layer('encoder').output
    return keras.models.Model(inputs=model.input, outputs=encoder_output)

# test data load
def load_data():
    path = 'PATH'
    list_ = os.listdir(path)
    normal = np.load(os.path.join(path, list_[-1], "NORMAL.npy"))
    apnea = np.load(os.path.join(path, list_[-1], "APNEA.npy"))
    X = np.concatenate((normal, apnea), axis=0)
    y = np.array([0] * normal.shape[0] + [1] * apnea.shape[0])
    print(f"Data shape: {X.shape}, {y.shape}")
    return X, y

# selective prediction loss
def selective_loss(y_true, y_pred):
    lambda_ = 2.0e2
    c = 0.92  # coverage
    gt = tf.keras.backend.repeat_elements(y_pred[:,-1:], 2, axis=1) * y_true
    pred = y_pred[:,:-1]
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=pred))
    coverage_loss = lambda_ * tf.keras.backend.maximum(-tf.reduce_mean(y_pred[:,-1]) + c, 0) ** 2
    return ce_loss + coverage_loss

# threshold 0.90 - 0.98
def evaluate_model(model, X, y, threshold=0.93):
    results = model.predict(X)
    selection_scores = results[1][:, -1]
    selected_indices = np.where(selection_scores >= threshold)[0]
    
    X_selected = X[selected_indices]
    y_selected = y[selected_indices]
    y_pred = (results[0][selected_indices, -1] >= 0.5).astype(int)
    
    accuracy = np.mean(y_selected == y_pred)
    rejection_rate = 1 - len(selected_indices) / len(X)
    
    # classification result
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Rejection rate: {rejection_rate:.4f}")
    print(classification_report(y_selected, y_pred, target_names=['normal', 'apnea']))
    
    # auc curve
    fpr, tpr, thresholds = roc_curve(y_selected, results[0][selected_indices, -1])
    auc_score = roc_auc_score(y_selected, results[0][selected_indices, -1])
    print(f"AUC: {auc_score:.4f}")
    
    plt.figure()
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    
    # confusion matrix
    cm = confusion_matrix(y_selected, y_pred)
    plot_confusion_matrix(conf_mat=cm)
    plt.tight_layout()
    plt.show()

def main():
    X, y = load_data()
    model = keras.models.load_model("model.h5",
                                    custom_objects={'cce': tf.keras.losses.BinaryCrossentropy(),
                                                    'selective_loss': selective_loss})
    evaluate_model(model, X, y)

if __name__ == "__main__":
    main()