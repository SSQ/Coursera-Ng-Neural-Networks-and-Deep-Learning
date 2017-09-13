# Goal
Build a logistic regression model, structured as a shallow neural network
# File Description
- `.ipynb` file is the solution of Week 2 program assignment 1
  - `Python+Basics+With+Numpy+v3.ipynb`
  - `Logistic+Regression+with+a+Neural+Network+mindset+v3.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `Python+Basics+With+Numpy+v3.html`
  - `Logistic+Regression+with+a+Neural+Network+mindset+v3.html`

# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- Logistic Regression

# Implementation
- Initialize the parameters of the model
- Learn the parameters for the model by minimizing the cost  
- Use the learned parameters to make predictions (on the test set)
- Analyse the results and conclude

# Implementation in detail

**What you need to remember**:
  - Common steps for pre-processing a new dataset are:
      - Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
      - Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
      - "Standardize" the data

**Implementation**:
  - Write a function `sigmoid(z)` 
  - Write a function `initialize_with_zeros(dim)` return `w, b`
  - Write a function `propagate(w, b, X, Y)` return `grads, cost`
  - Write a function `optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False)` return `params, grads, costs`
  - Write a function `predict(w, b, X)` return `Y_prediction`

**What to remember**: You've implemented several functions that:
  - Initialize (w,b)
  - Optimize the loss iteratively to learn parameters (w,b):
      - computing the cost and its gradient
      - updating the parameters using gradient descent
  - Use the learned (w,b) to predict the labels for a given set of examples

**Implementation**:
  - Write a function `model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False)` return model `d`

**What to remember** from this assignment:
  - Preprocessing the dataset is important.
  - You implemented each function separately: initialize(), propagate(), optimize(). Then you built a model().
  - Tuning the learning rate (which is an example of a "hyperparameter") can make a big difference to the algorithm. You will see more examples of this later in this course!

**Bibliography**:
  - http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
  - https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c
