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
    return 1 / (1 + np.exp(-output))
    
def sigmoid_derivative(output):
# Pochodna sigmoidy jako wynik: s * (1 - s)
    s = sigmoid(output)
    return s * (1 - s)

def leaky_ReLU_derivative(output):
    return np.where(output > 0, 1, 0.01)

def Categorical_Cross_Entropy(y_expected, y_output):
    output_log = np.log(y_output)
    return (-np.dot(y_expected, output_log.T))

def Mean_Square_Error(y_expected, y_output):
    return np.mean((y_output - y_expected)**2)