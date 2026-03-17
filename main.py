import numpy as np

def l_relu(x):
    return np.where(x>0, x, -0.00001*x)
    
def sigmoid_func(output):
    return 1 / (1 + np.exp(-output))

def softmax(output):
    exp_output = np.exp(output)
    exp_sum = np.sum(exp_output)
    return exp_output / exp_sum


class Layer_Dense:
    def __init__(self, input_size, output_size):
        self.weight = 0.1*np.random.rand(output_size, input_size)
        self.biases = 0.1*np.random.rand(output_size)
#weight[i,j] is a weight betweenn previous layer i-th neuron and current
#layer j-th neuron

    def forward(self, inputs, activation = None):
        self.output = np.dot(inputs, self.weight.T) + self.biases
        if activation == 'l_relu':
            self.output = l_relu(self.output)
        elif activation == 'sigmoid':
            self.output = self.sigmoid()
        return self.output
    def sigmoid(self):
        self.sigmoid_output = 1 / (1 + np.exp(-self.output))
        return self.sigmoid_output



X = np.random.rand(2)
y_expected = [0,1]
print(X, "\n")

layer_1 = Layer_Dense(2,4)  
layer_1.forward(X, activation = 'sigmoid')

layer_2 = Layer_Dense(4,4)
layer_2.forward(layer_1.output, activation = 'sigmoid')

y_layer = Layer_Dense(4,2)
y_output = y_layer.forward(layer_2.output)


print(softmax(y_output))