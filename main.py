import numpy as np

def leaky_ReLU(output):
    return np.where(output>0, output, 0.01 *output)

def softmax(output):
    exp_output = np.exp(output)
    exp_sum = np.sum(exp_output)
    return exp_output / exp_sum

'''
def Categorical_Cross_Entropy(y_expected, y_output):
    output_log = np.log(y_output)
    return (-np.dot(y_expected, output_log.T))
'''

def Mean_Square_Error(y_expected, y_output):
    return np.mean((y_output - y_expected)**2)


class Layer_Dense:
    def __init__(self, input_size, output_size):
        self.weight = 0.1*np.random.rand(output_size, input_size)
        self.biases = 0.1*np.random.rand(output_size)
#weight[i,j] is a weight betweenn j-th neuron and previous'
#layer i-th neuron
    def forward(self, inputs):
        self.inputs = inputs   
        self.output = leaky_ReLU(np.dot(inputs, self.weight.T) + self.biases)
        return self.output
    def backward(self, y_expected, y_output, learning_rate):
        #dLoss_dWeights = dLoss_dOuput * dReLU_dWeightedSum * dWeightedSum_dWeights
        dLoss_dOutput = 2 * (y_output - y_expected)                                 
        dRelu_WeightedSum = np.where(self.output>0, 1, 0.01)
        dWeightedSum_dWeights = self.inputs
        dLoss_dWeights = np.outer((dLoss_dOutput * dRelu_WeightedSum), dWeightedSum_dWeights.T)
        self.dLoss_dWeights = dLoss_dWeights
        self.weight -= (learning_rate * self.dLoss_dWeights)
        return self.weight

        
    
    def sigmoid(self):
        self.sigmoid_output = 1 / (1 + np.exp(-self.output))
        return self.sigmoid_output



X = np.random.rand(2)
y_expected = [0,1]
learning_rate = 0.01
print(X, "\n")
'''
layer_1 = Layer_Dense(2,4)  
layer_1.forward(X)
#print(layer_1.output, "\n")

layer_2 = Layer_Dense(4,4)
layer_2.forward(layer_1.output)
#print(layer_2.output, "\n")

y_layer = Layer_Dense(4,2)
y_output = y_layer.forward(layer_2.output)

#loss_CCE = Categorical_Cross_Entropy(y_expected, y_output)
loss_MSE = Mean_Square_Error(y_expected, y_output) 
'''
layer_1 = Layer_Dense(input_size=2, output_size=4)
layer_2= Layer_Dense(input_size=4, output_size=4)
layer_3 = Layer_Dense(input_size=4, output_size=2)
for epoch in range(180):
    hidden1_output = layer_1.forward(X)
    hidden2_output = layer_2.forward(hidden1_output)
    output = layer_3.forward(hidden2_output)
    if epoch %10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
print("Final weights: ", layer.weight)