import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tensorflow import keras

# data split
def load_and_split_data():
    X_DATASET, Y_DATASET = data_load()
    X_train, x, y_train, y = train_test_split(X_DATASET, Y_DATASET, test_size=0.3, random_state=42)
    print(f"Sample count: {len(x)}")
    return x, y

# tsne visualization
def perform_tsne(data, n_components=2):
    tsne = TSNE(n_components=n_components)
    return tsne.fit_transform(data)

# tsne plot
def plot_tsne(cluster, labels, title, cifar, colors=None, markers=None):
    plt.figure(figsize=(7, 7))
    plt.title(title)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    
    for i, label in enumerate(cifar):
        idx = np.where(labels == i)
        color = colors[i] if colors else None
        marker = markers[i] if markers else '.'
        plt.scatter(cluster[idx, 0], cluster[idx, 1], marker=marker, label=label, color=color, s=16)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# latent vector extraction
def create_new_model(model):
    model.layers[5]._name = "LSTM_RESULT"
    new_output = model.get_layer('LSTM_RESULT').output
    new_model = keras.models.Model(inputs=model.input, outputs=new_output)
    new_model.summary()
    return new_model

# rejection algorithm
def process_rejection(model, x, y, pro):
    final_result = model.predict(x, verbose=1)
    
    asult = np.array([i[-1] for i in final_result[1]])
    index = np.where(asult >= pro)
    
    x_label = []
    y_label = []
    for i, factor in enumerate(final_result[0]):
        x_label.append(x[i])
        y_label.append(y[i] if i in index[0] else 2)
    
    return np.array(x_label), np.array(y_label)

def main():
    x, y = load_and_split_data()
    
    # Before classification
    x_re = x.reshape(-1, 150*8)
    cluster = perform_tsne(x_re)
    plot_tsne(cluster, y, "Before Classification", ['normal', 'apnea'])
    
    # After classification
    new_model = create_new_model(model)
    classification_result = new_model.predict(x, verbose=1)
    cluster = perform_tsne(classification_result)
    plot_tsne(cluster, y, "After Classification", ['normal', 'apnea'], colors=['blue', 'red'])
    
    # With rejection
    x_label, y_label = process_rejection(model, x, y, pro)
    classification_result = new_model.predict(x_label, verbose=1)
    cluster = perform_tsne(classification_result)
    plot_tsne(cluster, y_label, "After Classification Including Rejection", 
              ['normal', 'apnea', 'rejection'], 
              colors=['cornflowerblue', 'lightcoral', 'green'],
              markers=['.', '.', '*'])

if __name__ == "__main__":
    main()

# selective risk extraction
def selective_risk_at_coverage(pred, y_test, coverage):
    # pred is auxiliary output
    sr = np.max(pred, axis=1)
    sr_sorted = np.sort(sr)
    threshold = sr_sorted[pred.shape[0] - int(coverage * pred.shape[0])]
    covered_idx = sr > threshold #### boolean value
    selective_acc = np.mean(np.argmax(pred[covered_idx], 1) == np.argmax(y_test[covered_idx],1))
    return selective_acc
 
# calculate coverage violation
def coverage_violation(y_test, target_coverage):
    violation = []
    for i in y_test:
        violation.append(abs(i-target_coverage))
    #print(violation)
    coverage_violation_result = np.mean(violation)
    coverage_mean = np.mean(y_test)
    coverage_sd = np.std(y_test)
    return coverage_violation_result, coverage_mean, coverage_sd