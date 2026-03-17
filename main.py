import numpy as np

def l_relu(x):
    a = np.maximum(0,x)
    if a==0:
        return -0.0001*x
    else:
        return x
    
def sigmoid_func(output):
        for i in range(len(output)):
            sigmoid_output = (1 / (1 + np.exp(-output)))
        return sigmoid_output


'''
inputs = np.random.rand(2)
weight_matrix = np.random.rand(3,3)
biases = np.random.rand(3)

outputs = np.dot(inputs, weight_matrix)+biases
for i in range(len(outputs)):
    outputs[i] = l_relu(outputs[i])

outputs = normalize(outputs)
print(outputs)
'''
class Layer_Dense:
    def __init__(self, input_size, output_size):
        self.weight = np.random.rand(output_size, input_size)
        self.biases = np.random.rand(output_size)
#weight[i,j] is a weight betweenn previous layer i-th neuron and current
#layer j-th neuron

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight.T) + self.biases
    def sigmoid(self):
        self.sigmoid_output = 1 / (1 + np.exp(-self.output))
        return self.sigmoid_output



X = np.random.rand(2)
y_expected = [0,1]
print(X, "\n")
layer1 = Layer_Dense(2,4)  
layer1.forward(X)
layer1.sigmoid()
layer2 = Layer_Dense(4,4)
layer2.forward(layer1.output)
layer2.sigmoid()
y = Layer_Dense(4,2)
y.forward(layer2.output)
y.sigmoid()
print(y.sigmoid_output,"\n")
Z = np.exp(y.output)
print(Z)

def softmax(output):
    exp_sum = 0.0
    probability = []
    for i in range(len(output)):
        exp_sum += (float)(np.exp(output[i]))
        probability.append((float)((np.exp(output[i]))/(exp_sum)))
    return probability

print(softmax(y.sigmoid_output))