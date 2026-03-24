import numpy as np

def leaky_ReLU(output):
    return np.where(output>0, output, 0.01 *output)

def softmax(output):
    #zmieniony softmax, gdybys mial duze wartosci w porpzenim, dostalbys inf, ten softmax bedzie obslugiwal dowolny batch
    shifted = output - np.max(output, axis=1, keepdims=True)
    exp_output = np.exp(shifted)
    exp_sum = np.sum(exp_output, axis=1, keepdims=True)
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
        self.z = np.dot(inputs, self.weight.T) + self.biases #zeby przechowywwalo sume wazona przed aktywacja, lepiej zrobic osobna zmienna do tego 
        self.output = leaky_ReLU(self.z)
        return self.output
    
    def backward(self, grad_from_next_layer, learning_rate):
        #dLoss_dWeights = dLoss_dOuput * dReLU_dWeightedSum * dWeightedSum_dWeights
        batch_size = self.inputs.shape[0]
        #tu wazna zmiana: wprowadzamy batch size. gdy siec sie uczy, to aktualizujemy wagi i biasy na podstawie sredniej z batcha, a nie pojedynczego przykladu, 
        #do sieci nie wrzucamy tylko jednej paczki danych ale zestaw paczek, czyli batch, potem oliczna ajest srednia z tego batcha
        #bo obliczneia zajelyby duzo czasu gdyby aktualizowac wagi pojedynczo. tym sam,ym usredniamy potem gradienty
        #od wszystkich probek w batchu, aktualizacja nie zalezy od wielkosci batcha, dlatego bedziemy rpzez batch size dzielic
        dReLU_WeightedSum = np.where(self.z > 0, 1, 0.01)
        grad_z = grad_from_next_layer * dReLU_WeightedSum

        self.dLoss_dWeights = np.dot(grad_z.T, self.inputs) / batch_size
        self.dLoss_dBiases = np.sum(grad_z, axis=0) / batch_size

        grad_from_prev_layer = np.dot(grad_z, self.weight)
        self.weight -= learning_rate * self.dLoss_dWeights
        self.biases -= learning_rate * self.dLoss_dBiases #boasy tez sie aktualizuje, nie tylko wagi 

        return grad_from_prev_layer #wczesniej, jesli zwracales self.weight, to zwracalo wagi, nie gradient
        #propagacja wsteczna w uczeniu wymaga przekazania gradientu do warstwy poprzeniej, inaczje nie mozna trenowac 
        #wiecej niz jednej warstwy  

        
    
    def sigmoid(self):
        self.sigmoid_output = 1 / (1 + np.exp(-self.output))
        return self.sigmoid_output



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Batch 4 probki
y_expected = np.array([[0], [1], [1], [0]]) #Batch 4 probki 1 output
learning_rate = 0.01
#print(X, "\n")
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
layer_3 = Layer_Dense(input_size=4, output_size=1) #wyjscie z 2 na 1 neuron, na potrzeby klasyfikajji binarnej
print("First weights of third layer: ", layer_3.weight)
for epoch in range(1800):
    hidden1_output = layer_1.forward(X)
    hidden2_output = layer_2.forward(hidden1_output)
    output = layer_3.forward(hidden2_output)
    
    #ponizej definicja loss, bo nie byl liczony
    loss = Mean_Square_Error(y_expected, output)
    #backward
    grad3 = 2 * (output - y_expected)
    grad3 = layer_3.backward(grad3, learning_rate)
    grad2 = layer_2.backward(grad3, learning_rate)
    grad1 = layer_1.backward(grad2, learning_rate)
    if epoch %100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
print("Final weights of third layer: ", layer_3.weight)