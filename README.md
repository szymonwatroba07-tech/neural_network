# Neural_network from scratch 

## XOR solution 
This repository features a lightweight yet robust implementation of a Multi-Layer Perceptron (MLP) built entirely from the ground up using NumPy. By bypassing high-level frameworks like TensorFlow or PyTorch, this project offers a transparent look at the underlying mathematics of deep learning, specifically designed to conquer the non-linear complexity of the XOR logic gate.

## Technical Architecture 
The core of the engine resides in the Layer_Dense class, which manages the lifecycle of a fully connected layer. To ensure stable training from the first epoch, the model utilizes He Initialization, scaling weights by sqrt(2/n)​ to maintain variance across layers. This is paired with Leaky ReLU activations in the hidden layers to prevent the "dying neuron" problem by allowing a small gradient (0.01) for negative inputs. The output layer employs a Sigmoid function, σ(z)=1+e−z1​, effectively squashing raw scores into a (0,1) probability range for binary classification.

The learning process is driven by the Chain Rule of calculus during the backpropagation phase. The network calculates the gradient of the Mean Squared Error (MSE) loss relative to the output
