# Goal
Planar data classification with a hidden layer
# File Description
- `.ipynb` file is the solution of Week 2 program assignment 1
  - `Planar+data+classification+with+one+hidden+layer+v3.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `Planar+data+classification+with+one+hidden+layer+v3.html`

# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- Shallow Neural Network

# Implementation
- Build a 2-class classification complete neural network with a hidden layer
- Make a good use of a non-linear unit, such as tanh
- Compute the cross entropy loss
- Implemented forward propagation and backpropagation, and trained a neural network
- See the impact of varying the hidden layer size, including overfitting.

# Implementation in detail

**What you need to remember**:
  - Common steps for pre-processing a new dataset are:
      - Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
      - Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
      - "Standardize" the data

**Implementation**:
1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
  - Write a function `layer_sizes(X, Y)` `return (n_x, n_h, n_y)` 
2. Initialize the model's parameters
  - Write a function `initialize_parameters(n_x, n_h, n_y)` `return parameters`
3. Loop:
    - Implement forward propagation
      - Write a function `forward_propagation(X, parameters)` `return A2, cache`
    - Compute loss
      - Write a function `compute_cost(A2, Y, parameters)` `return cost`
    - Implement backward propagation to get the gradients
      - Write a function `backward_propagation(parameters, cache, X, Y)` `return grads`
    - Update parameters (gradient descent)
      - Write a function `update_parameters(parameters, grads, learning_rate = 1.2)` `return parameters`
4. Build your neural network model 
  - Write a function `nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False)` `return parameters`
5. Predictions
  - Write a function `predict(parameters, X)` `return predictions`

# Reference:
http://scs.ryerson.ca/~aharley/neural-networks/
http://cs231n.github.io/neural-networks-case-study/

