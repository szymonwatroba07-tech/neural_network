import numpy as np

def leaky_ReLU(output):
    return np.where(output>0, output, 0.01 *output)

def softmax(output):
    #zmieniony softmax, gdybys mial duze wartosci w porpzenim, dostalbys inf, ten softmax bedzie obslugiwal dowolny batch
    shifted = output - np.max(output, axis=1, keepdims=True)
    exp_output = np.exp(shifted)
    exp_sum = np.sum(exp_output, axis=1, keepdims=True)
    return exp_output / exp_sum

def sigmoid(output):
    return np.where(output >= 0, 1 / (1 + np.exp(-output)),  np.exp(output) / (1 + np.exp(output)))
    
def sigmoid_derivative(output):
# Pochodna sigmoidy jako wynik: s * (1 - s)
    s = sigmoid(output)
    return s * (1 - s)

def leaky_ReLU_derivative(output):
    return np.where(output > 0, 1, 0.01)

def Categorical_Cross_Entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    row_sums = np.sum(y_true * np.log(y_pred), axis=1)
    
    return -np.mean(row_sums)
    
def Mean_Square_Error(y_expected, y_output):
    return np.mean((y_output - y_expected)**2)

def load_images(images_path):
    with open(images_path, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
        images = images.reshape(-1, 784)
        return images

def load_labels(labels_path):
    with open(labels_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset = 8)
        return labels
    
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10)) # Tworzy macierz zer 60000x10
    one_hot_Y[np.arange(Y.size), Y] = 1 # Wstawia 1 tam, gdzie jest dana cyfra
    return one_hot_Y