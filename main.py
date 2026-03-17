import numpy as np

def l_relu(x):
    a = np.maximum(0,x)
    if a==0:
        return -0.0001*x
    else:
        return x

def normalize(outputs):
    max = np.max(outputs)
    return outputs/max


inputs = np.random.rand(3)
weight_matrix = np.random.rand(3,3)
biases = np.random.rand(3)

outputs = np.dot(inputs, weight_matrix)+biases
for i in range(len(outputs)):
    outputs[i] = l_relu(outputs[i])

outputs = normalize(outputs)
print(outputs)

class Layer_Dense:
    def __init__(self, n_inputs, n_outputs):
        self.weight = np.random.rand(n_outputs, n_inputs)
        self.biases = np.random.rand(n_outputs)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight.T) + self.biases

layer_1 = Layer_Dense(3,3)
print(layer_1.weight, layer_1.biases)
layer_1.forward(3)
outputs = normalize(layer_1.output)
print(outputs)
layer_2 = Layer_Dense(len(outputs), 3)
layer_2.forward(len(outputs))
outputs = normalize(layer_2.output)
print(outputs)