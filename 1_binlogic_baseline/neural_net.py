import numpy as np
import matplotlib.pyplot as plt


def sigmoid(Z):
    return 1/(1+np.exp(-Z)), Z


def relu(Z):
    return np.maximum(0,Z), Z


def leaky_relu(Z, alpha=0.01):
    return np.maximum(alpha*Z, Z), Z


def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)[0]
    return dA * sig * (1 - sig)


def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;


def leaky_relu_backward(dA, Z, alpha=0.01):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = alpha;
    return dZ;


def initialize_parameters_deep(layer_dims, seed, weights=None, weight_multiplier=0.1, weight_addition=0):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(seed)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network
    
    if weights is None:
        for l in range(1, L):
            parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1])*weight_multiplier + weight_addition
            parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    else:
        for l in range(1, L):
            parameters[f'W{l}'] = np.array(weights[l-1][:, 1:])
            parameters[f'b{l}'] = np.array(weights[l-1][:, 0]).reshape(-1, 1)
    
    return parameters


def linear_forward(A, W, b):
    """
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    # print(f'W shape: {W.shape}, A shape: {A.shape}, b shape: {b.shape}')
    Z = np.dot(W, A) + b.reshape(-1, 1)
    cache = (A, W, b)   
    return Z, cache

    
def linear_activation_forward(A_prev, W, b, activation):
    """
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "leaky_relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = leaky_relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters, hidden_act='relu'):
    """    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
        
    for l in range(1, L):
        A_prev = A 
        W, b = parameters['W' + str(l)], parameters['b' + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, hidden_act)
        caches.append(cache)
    
    W, b = parameters['W' + str(L)], parameters['b' + str(L)]
    AL, cache = linear_activation_forward(A, W, b, 'sigmoid')
    caches.append(cache)
            
    return AL, caches


def compute_cost_entropy(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    cost = -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))    
    cost = np.squeeze(cost)
    return cost


def compute_cost_mse(AL, Y):
    return np.mean((AL - Y) ** 2)


def compute_class_error(AL, Y):
    return np.sum(abs(classify(AL) - Y))


def linear_backward(dZ, cache):
    """
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, np.transpose(A_prev)) / m
    db = np.mean(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(np.transpose(W), dZ)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "leaky_relu":
        dZ = leaky_relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, hidden_act='relu', criterion='entropy'):
    """    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    if criterion == 'entropy':
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    elif criterion == 'mse':
        dAL = 2*(AL - Y)
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(grads["dA" + str(l+1)], current_cache, hidden_act)

    return grads


def update_parameters(parameters, grads, learning_rate):
    """    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    L = len(parameters) // 2 # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return parameters


def classify(activations):
    predictions = np.where(activations > 0.5, 1, 0)
    return predictions


class MyNetwork:
    def __init__(self, dim_list, seed=0, weights=None, weight_multiplier=0.1, hidden_act='relu', weight_addition=0):
        self.seed = seed
        self.hidden_act = hidden_act
        self.parameters = initialize_parameters_deep(dim_list, seed, weights=weights, weight_multiplier=weight_multiplier, weight_addition=weight_addition)

    def fit(self, X, Y, learning_rate=0.0075, num_iterations=3000, criterion='entropy', print_cost=False, plot_costs=True):
        entropy_costs = []
        misclass = []
        mses = []
        for i in range(0, num_iterations):
            AL, caches = L_model_forward(X, self.parameters, hidden_act=self.hidden_act)
            
            if criterion == 'entropy':
                cost = compute_cost_entropy(AL, Y)
            elif criterion == 'mse':
                cost = compute_cost_mse(AL, Y)

            grads = L_model_backward(AL, Y, caches, hidden_act=self.hidden_act, criterion=criterion)
            self.parameters = update_parameters(self.parameters, grads, learning_rate)

            # Print the cost every 1000 training example
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            
            if criterion == 'entropy':
                entropy_costs.append(cost)
                mses.append(compute_cost_mse(AL, Y))
            elif criterion == 'mse':
                entropy_costs.append(compute_cost_entropy(AL, Y))
                mses.append(cost)
            
            misclass.append(compute_class_error(AL, Y)/X.shape[1])

        if plot_costs:
            # plot the cost
            plt.plot(np.squeeze(entropy_costs), label='Entropy')
            plt.plot(np.squeeze(misclass), label='% of Misclassifications')
            plt.plot(np.squeeze(mses), label='MSE')
            plt.ylabel('Cost')
            plt.xlabel('Iterations')
            plt.title(f"activation={self.hidden_act}, lr={learning_rate}, num_iter={num_iterations}, seed={self.seed}")
            plt.legend()
            plt.savefig(f'imgs/error_act{self.hidden_act}_seed{self.seed}')
            plt.show()
    
        print('Training results:')
        print(f'Cross-Entropy: {entropy_costs[-1]}')
        print(f'MSE: {mses[-1]}')
        print(f'Misclassification count: {int(misclass[-1] * X.shape[1])}/{X.shape[1]}')
        print()
        
        return entropy_costs[-1], mses[-1], misclass[-1]
        
    def predict(self, X):
        activations, caches = L_model_forward(X, self.parameters)
        predictions = np.where(activations > 0.5, 1, 0)
        return predictions
    
    def score(self, AL, Y):
        cross_entropy = compute_cost_entropy(AL, Y)
        print('Training results:')
        print(f'Cross-Entropy: {cross_entropy}')
        print(f'MSE: {compute_cost_mse(AL, Y)}')
        print(f'Misclassification count: {compute_class_error(AL, Y)}/{X.shape[1]}') 
