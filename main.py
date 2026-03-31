import numpy as np
import matplotlib.pyplot as plt

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

'''
def Categorical_Cross_Entropy(y_expected, y_output):
    output_log = np.log(y_output)
    return (-np.dot(y_expected, output_log.T))
'''

def Mean_Square_Error(y_expected, y_output):
    return np.mean((y_output - y_expected)**2)


class Layer_Dense:
    def __init__(self, input_size, output_size, activation = "leaky_ReLU"):
        self.weight = 0.1*np.random.randn(output_size, input_size) #randn bo wagi moga byc ujemne
        #biasy jako macierz dla poprawnego broadcastingu, zeby mozna bylo dodawac do kazdego wiersza, a nie tylko do pierwszego
        self.biases = np.zeros((1, output_size))
        self.activation = activation
#weight[i,j] is a weight betweenn j-th neuron and previous'
#layer i-th neuron
    def forward(self, inputs):
        self.inputs = inputs   
        self.z = np.dot(inputs, self.weight.T) + self.biases #zeby przechowywwalo sume wazona przed aktywacja, lepiej zrobic osobna zmienna do tego 
        if self.activation == "leaky_ReLU":
            self.output = leaky_ReLU(self.z)
        elif self.activation == "sigmoid":
            self.output = sigmoid(self.z)
        return self.output
    
    def backward(self, grad_from_next_layer, learning_rate):
        #dLoss_dWeights = dLoss_dOuput * dReLU_dWeightedSum * dWeightedSum_dWeights
        batch_size = self.inputs.shape[0]
        #tu wazna zmiana: wprowadzamy batch size. gdy siec sie uczy, to aktualizujemy wagi i biasy na podstawie sredniej z batcha, a nie pojedynczego przykladu, 
        #do sieci nie wrzucamy tylko jednej paczki danych ale zestaw paczek, czyli batch, potem oliczna ajest srednia z tego batcha
        #bo obliczneia zajelyby duzo czasu gdyby aktualizowac wagi pojedynczo. tym sam,ym usredniamy potem gradienty
        #od wszystkich probek w batchu, aktualizacja nie zalezy od wielkosci batcha, dlatego bedziemy rpzez batch size dzielic
        if self.activation == "leaky_ReLU":
            d_activation = leaky_ReLU_derivative(self.z)
        elif self.activation == "sigmoid":
            d_activation = self.output * (1-self.output)
        grad_z = grad_from_next_layer * d_activation

        self.dLoss_dWeights = np.dot(grad_z.T, self.inputs) / batch_size
        self.dLoss_dBiases = np.sum(grad_z, axis=0, keepdims=True) / batch_size

        grad_from_prev_layer = np.dot(grad_z, self.weight)
        self.weight -= learning_rate * self.dLoss_dWeights
        self.biases -= learning_rate * self.dLoss_dBiases #boasy tez sie aktualizuje, nie tylko wagi 

        return grad_from_prev_layer #wczesniej, jesli zwracales self.weight, to zwracalo wagi, nie gradient
        #propagacja wsteczna w uczeniu wymaga przekazania gradientu do warstwy poprzeniej, inaczje nie mozna trenowac 
        #wiecej niz jednej warstwy  

    



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Batch 4 probki
y_expected = np.array([[0], [1], [1], [0]]) #Batch 4 probki 1 outputl
learning_rate = 0.1
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

# Architektura: wejście 2 -> ukryta 4 -> ukryta 4 -> wyjście 1 (Sigmoid)
layer_1 = Layer_Dense(input_size=2, output_size=4, activation="leaky_ReLU")
layer_2= Layer_Dense(input_size=4, output_size=4, activation="leaky_ReLU")
layer_3 = Layer_Dense(input_size=4, output_size=1, activation="sigmoid") #sigmoid na ywjscie, nie moze byc relu bo wynik bedzie bezsensowny
print("First weights of third layer: ", layer_3.weight)
loss_arr = []
for epoch in range(1000):
    hidden1_output = layer_1.forward(X)
    hidden2_output = layer_2.forward(hidden1_output)
    output = layer_3.forward(hidden2_output)
    
    #ponizej definicja loss, bo nie byl liczony
    loss = Mean_Square_Error(y_expected, output)
    #tabela strat do wykresu
    loss_arr.append(loss)
    #backward
    grad = 2 * (output - y_expected) / output.shape[0]
    
    grad = layer_3.backward(grad, learning_rate)
    grad = layer_2.backward(grad, learning_rate)
    grad = layer_1.backward(grad, learning_rate)
    if epoch %100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
print("Final weights of third layer: ", layer_3.weight)

plt.figure(figsize=(10, 6))
plt.plot(loss_arr, label='Training Loss')

plt.title('Funkcja straty (MSE) w czasie trenowania')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()