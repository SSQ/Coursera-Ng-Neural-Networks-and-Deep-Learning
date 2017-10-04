# Deep Neural Network - Application
# Goal
Deep Neural Network for Image Classification: Application
# File Description
- `.ipynb` file is the solution of Week 4 program assignment 2
  - `Deep+Neural+Network+-+Application+v3.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `Deep+Neural+Network+-+Application+v3.html`

# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- Deep Neural Network

# Implementation
- Learn how to use all the helper functions you built in the previous assignment to build a model of any structure you want.
- Experiment with different model architectures and see how each one behaves.
- Recognize that it is always easier to build your helper functions before attempting to build a neural network from scratch.

# Implementation in detail

## **Deep Learning methodology**
1. Initialize parameters / Define hyperparameters
2. Loop for num_iterations:
    a. Forward propagation
    b. Compute cost function
    c. Backward propagation
    d. Update parameters (using parameters, and grads from backprop) 
4. Use trained parameters to predict labels

## Two-layer neural network
```python
def initialize_parameters(n_x, n_h, n_y):
    ...
    return parameters 
def linear_activation_forward(A_prev, W, b, activation):
    ...
    return A, cache
def compute_cost(AL, Y):
    ...
    return cost
def linear_activation_backward(dA, cache, activation):
    ...
    return dA_prev, dW, db
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
```

```python
### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
```

```python
# GRADED FUNCTION: two_layer_model

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """    
    return parameters
```
## Accuracy
- train 1.0
- test 0.72
## L-layer Neural Network
```python
def initialize_parameters_deep(layer_dims):
    ...
    return parameters 
def L_model_forward(X, parameters):
    ...
    return AL, caches
def compute_cost(AL, Y):
    ...
    return cost
def L_model_backward(AL, Y, caches):
    ...
    return grads
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
```

```python
### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
```

```python
# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """    
    return parameters
```
## Accuracy
- train 0.99
- test 0.8
# References:
- for auto-reloading external module: http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
- [What does -1 mean in numpy reshape?](https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape)
- [numpy.random.randn](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.randn.html)
- [what the role of keepdims in python](https://stackoverflow.com/questions/40927156/what-the-role-of-keepdims-in-python)
