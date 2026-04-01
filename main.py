import numpy as np
import matplotlib.pyplot as plt
import functions as func


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
            self.output = func.leaky_ReLU(self.z)
        elif self.activation == "sigmoid":
            self.output = func.sigmoid(self.z)
        return self.output
    
    def backward(self, grad_from_next_layer):
        #dLoss_dWeights = dLoss_dOuput * dReLU_dWeightedSum * dWeightedSum_dWeights
        batch_size = self.inputs.shape[0]
        #tu wazna zmiana: wprowadzamy batch size. gdy siec sie uczy, to aktualizujemy wagi i biasy na podstawie sredniej z batcha, a nie pojedynczego przykladu, 
        #do sieci nie wrzucamy tylko jednej paczki danych ale zestaw paczek, czyli batch, potem oliczna ajest srednia z tego batcha
        #bo obliczneia zajelyby duzo czasu gdyby aktualizowac wagi pojedynczo. tym sam,ym usredniamy potem gradienty
        #od wszystkich probek w batchu, aktualizacja nie zalezy od wielkosci batcha, dlatego bedziemy rpzez batch size dzielic
        if self.activation == "leaky_ReLU":
            d_activation = func.leaky_ReLU_derivative(self.z)
        elif self.activation == "sigmoid":
            d_activation = self.output * (1-self.output)
        grad_z = grad_from_next_layer * d_activation

        self.dLoss_dWeights = np.dot(grad_z.T, self.inputs)
        self.dLoss_dBiases = np.sum(grad_z, axis=0, keepdims=True)

        grad_from_prev_layer = np.dot(grad_z, self.weight)

        return grad_from_prev_layer #wczesniej, jesli zwracales self.weight, to zwracalo wagi, nie gradient
        #propagacja wsteczna w uczeniu wymaga przekazania gradientu do warstwy poprzeniej, inaczje nie mozna trenowac 
        #wiecej niz jednej warstwy  

class Adam_Optimizer:
    def __init__(self, layers_matrix, learning_rate = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, decay_rate = 0.01):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.decay_rate = decay_rate
        
        self.m_weights = ([np.zeros_like(layer.weight) for layer in layers_matrix])  #inicjalizacja matryc m i v (wypelnienie zerami)
        self.v_weights = ([np.zeros_like(layer.weight) for layer in layers_matrix])
        self.m_biases = ([np.zeros_like(layer.biases) for layer in layers_matrix])
        self.v_biases = ([np.zeros_like(layer.biases) for layer in layers_matrix])

    def step(self, layer_matrix):
        self.t +=1
        for i, layer in enumerate(layer_matrix):
            
            self.m_weights[i] = self.beta_1 * self.m_weights[i] + (1-self.beta_1) * (layer.dLoss_dWeights)
            self.v_weights[i] = self.beta_2 * self.v_weights[i] + (1-self.beta_2) * (layer.dLoss_dWeights)**2

            self.m_biases[i] = self.beta_1 * self.m_biases[i] + (1-self.beta_1) * (layer.dLoss_dBiases)
            self.v_biases[i] = self.beta_2 * self.v_biases[i] + (1-self.beta_2) * (layer.dLoss_dBiases)**2

            # Bias-corrected estimates for weights
            m_w_hat = self.m_weights[i] / (1 - self.beta_1 ** self.t)
            v_w_hat = self.v_weights[i] / (1 - self.beta_2 ** self.t)
        
            # Bias-corrected estimates for biases
            m_b_hat = self.m_biases[i] / (1 - self.beta_1 ** self.t)
            v_b_hat = self.v_biases[i] / (1 - self.beta_2 ** self.t)
        
            layer.weight -= (self.learning_rate * m_w_hat) / (np.sqrt(v_w_hat) + self.epsilon) - layer.weight * self.decay_rate * self.learning_rate
            layer.biases -= (self.learning_rate * m_b_hat) / (np.sqrt(v_b_hat) + self.epsilon) - layer.biases * self.decay_rate * self.learning_rate
            

        
    
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Batch 4 probki
y_expected = np.array([[0], [1], [1], [0]]) #Batch 4 probki 1 outputl
learning_rate = 0.1


# Architektura: wejście 2 -> ukryta 4 -> ukryta 4 -> wyjście 1 (Sigmoid)
layer_1 = Layer_Dense(input_size=2, output_size=4, activation="leaky_ReLU")
layer_2= Layer_Dense(input_size=4, output_size=4, activation="leaky_ReLU")
layer_3 = Layer_Dense(input_size=4, output_size=1, activation="sigmoid") #sigmoid na ywjscie, nie moze byc relu bo wynik bedzie bezsensowny
print("First weights of third layer: ", layer_3.weight)


Adam = Adam_Optimizer([layer_1, layer_2, layer_3])
loss_arr = []
for epoch in range(250):
    hidden1_output = layer_1.forward(X)
    hidden2_output = layer_2.forward(hidden1_output)
    output = layer_3.forward(hidden2_output)
    
    #ponizej definicja loss, bo nie byl liczony
    loss = func.Mean_Square_Error(y_expected, output)
    #tabela strat do wykresu
    loss_arr.append(loss)
    #backward
    grad = 2 * (output - y_expected) / output.shape[0]
    
    grad = layer_3.backward(grad)
    grad = layer_2.backward(grad)
    grad = layer_1.backward(grad)

    Adam.step([layer_1, layer_2, layer_3])

    if epoch %25 == 0:
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
