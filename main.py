import numpy as np
import matplotlib.pyplot as plt
import functions as func

# Dane wejsciowe
train_images_path = r"C:\Users\szymo\neural_network\train-images.idx3-ubyte"
train_labels_path = r"C:\Users\szymo\neural_network\train-labels.idx1-ubyte"
test_images_path = r"C:\Users\szymo\neural_network\t10k-images.idx3-ubyte"
test_labels_path = r"C:\Users\szymo\neural_network\t10k-labels.idx1-ubyte"

train_images = func.load_images(train_images_path)
train_labels = func.load_labels(train_labels_path)

# Zagęszczenie warstw
class Layer_Dense:
    def __init__(self, input_size, output_size, activation="softmax"):
        # He Initialization dla Leaky ReLU, Xavier dla Sigmoid
        if activation == "leaky_ReLU":
            self.weight = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        else:
            self.weight = np.random.randn(output_size, input_size) * np.sqrt(1. / input_size)
            
        self.biases = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs   
        self.z = np.dot(inputs, self.weight.T) + self.biases #zeby przechowywwalo sume wazona przed aktywacja, lepiej zrobic osobna zmienna do tego 
        if self.activation == "leaky_ReLU":
            self.output = func.leaky_ReLU(self.z)
        elif self.activation == "softmax":
            self.output = func.softmax(self.z)
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
            grad_z = grad_from_next_layer * func.leaky_ReLU_derivative(self.z)
        elif self.activation == "softmax":
            grad_z = grad_from_next_layer
        elif self.activation == "sigmoid":
            grad_z = grad_from_next_layer * (self.output * (1 - self.output))

        self.dLoss_dWeights = np.dot(grad_z.T, self.inputs)
        self.dLoss_dBiases = np.sum(grad_z, axis=0, keepdims=True)

        grad_from_prev_layer = np.dot(grad_z, self.weight)

        return grad_from_prev_layer #wczesniej, jesli zwracales self.weight, to zwracalo wagi, nie gradient
        #propagacja wsteczna w uczeniu wymaga przekazania gradientu do warstwy poprzeniej, inaczje nie mozna trenowac 
        #wiecej niz jednej warstwy  

# Adam W
class Adam_Optimizer:
    def __init__(self, layers_matrix, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, decay_rate = 0.01):
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
        
            layer.weight *= (1 - self.decay_rate * self.learning_rate)
            layer.weight -= (self.learning_rate * m_w_hat) / (np.sqrt(v_w_hat) + self.epsilon)
            
            layer.biases -= (self.learning_rate * m_b_hat) / (np.sqrt(v_b_hat) + self.epsilon)

# Dane wejściowe           
X = train_images / 255.0
y_expected = func.one_hot(train_labels)
learning_rate = 0.01
batch_size = 128

# Architektura: wejście 784 -> ukryta 128 -> ukryta 64 -> wyjście 10 (Softmax)
layer_1 = Layer_Dense(input_size=784, output_size=128, activation="leaky_ReLU")
layer_2= Layer_Dense(input_size=128, output_size=64, activation="leaky_ReLU")
layer_3 = Layer_Dense(input_size=64, output_size=10, activation="softmax")

#Inicjalizacja otymalizatora
Adam = Adam_Optimizer([layer_1, layer_2, layer_3])
loss_arr = []
batch_size = 128
epochs = 40 

print(f"Y_expected shape: {y_expected.shape}, Sum of first row: {np.sum(y_expected[0])}")

for epoch in range(epochs):
    
    permutation = np.random.permutation(X.shape[0])
    X_shuffled = X[permutation]
    y_shuffled = y_expected[permutation]
    
    epoch_loss = 0
    steps = 0
    
    for i in range(0, X.shape[0], batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        
        # 1. FORWARD
        out1 = layer_1.forward(X_batch)
        out2 = layer_2.forward(out1)
        output = layer_3.forward(out2)
        
        # 2. LOSS (Categorical Cross Entropy)
        batch_loss = func.Categorical_Cross_Entropy(y_batch, output)
        epoch_loss += batch_loss
        steps += 1
        
        # 3. BACKWARD
        # Pochodna Softmax + CCE: (wyjście - prawda) / rozmiar_batcha
        grad = (output - y_batch) / X_batch.shape[0]
        
        grad = layer_3.backward(grad)
        grad = layer_2.backward(grad)
        grad = layer_1.backward(grad)
        
        # 4. OPTIMIZATION
        Adam.step([layer_1, layer_2, layer_3])
    
    avg_loss = epoch_loss / steps
    loss_arr.append(avg_loss)
    
    # Dodatkowy monitoring celności
    predictions = np.argmax(output, axis=1)
    targets = np.argmax(y_batch, axis=1)
    accuracy = np.mean(predictions == targets)
    
    print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Last Batch Acc: {accuracy*100:.2f}%")

print(f"Suma outputu: {np.sum(output[0])}")

# Wykres straty od epoki
plt.figure(figsize=(10, 6))
plt.plot(loss_arr, label='Training Loss')

plt.title('Funkcja straty (MSE) w czasie trenowania')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
