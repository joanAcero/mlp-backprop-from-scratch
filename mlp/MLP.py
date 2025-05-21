# Author: Joan Acero Pousa
# Date: May 2025

import numpy as np

class MLP:
    """ 
    This class implements a Multi-Layer Perceptron (MLP) neural network.
    It supports both classification and regression tasks.
    The backpropagation algorithm gradients can be checked using numerical gradient checking.
    The MLP can be trained using mini-batch gradient descent.
    """

    ALLOWED_HIDDEN_ACTIVATION_FUNCTIONS = ['ReLU'] 
    ALLOWED_TASKS = ['classification', 'regression']

    def __init__(self, hidden_layer_neurons: list = [], 
                 activation_functions: list = ['ReLU'],
                 task:str = 'classification',verbose:bool = True):
        """
        Builds a Multi Layer Perceptron architecture and initializes its weights.

        Input Parameters:
            - hidden_layer_neurons: Number of nodes for each hidden layer . 
            - activation_functions: Set the activation function for each layer.
            - task: task of the MLP.
            - verbose: If True, prints information while interacting with the MLP.
        """

        # Some checks on data provided
        if task not in self.ALLOWED_TASKS: raise ValueError(f"Task should be one of the following: {self.ALLOWED_TASKS}")
        if len(hidden_layer_neurons) < 0: raise ValueError(f"Number of hidden layers should be >= 0")

        # Set up the Multi-Layer Perceptron architecture: Number of layers, neurons per layer, activation functions.
        self.task = task
        self.num_hidden_layers = len(hidden_layer_neurons)
        self.num_layers = len(hidden_layer_neurons) +1
        self.layer_neurons = hidden_layer_neurons
        self.num_neurons = np.sum(self.layer_neurons)
        self.activation_functions = self._set_activation_functions(activation_functions)

        # Initialize the parameters
        self.weights = None
        self.biases = None

        # Set the verbose flag
        self.verbose = verbose

        # Initialize caches and gradient holders
        self.cache_Z = []  # To store pre-activations (Z values) for each layer
        self.X_batch_current = None # To store the X_batch used in the current forward pass
        self.dW = None     # To store weight gradients
        self.db = None     # To store bias gradients
        

        if self.verbose:
            print("#################################################################")
            print(f"Initialized the following Multi Layer Perceptron: \n"
                f"    -Number of hidden layers: {self.num_hidden_layers} \n"
                f"    -Number of neurons per layer: {self.layer_neurons} \n"
                f"    -Activation functions: {self.activation_functions} \n"
                f"    -Task: {self.task} \n")   
        
    def train(self, X,y, batch_size:int = 1, epochs:int = 1,learning_rate: float = 0.001,):
        """
        Trains the MLP for the given data, task, architecture and batch size.
        
        Input parameters:
            - X: Input data of shape (n_samples, n_features).
            - y: Target data of shape (n_samples, n_classes) for classification or (n_samples,) for regression.
            - batch_size (int): The number of data samples to process per batch during training.
            - epochs: Number of times the MLP will see the training data.
            - learning_rate: The learning rate for the training algorithm
        """
        
        X = np.asarray(X)
        y = np.asarray(y)

        if self.verbose: 
            print("#################################################################")
            print("Fitting the MLP for the given data:")
        
        # Initialize weights, biases, layers given the data structure
        self._fit(X,y, batch_size=batch_size)

        if self.verbose: 
            print("#################################################################")
            print("Training the MLP for the given data:")

        # Train the MLP
        self.learning_rate = learning_rate
        n_samples = X.shape[0]

        for epoch in range(epochs):
            if self.verbose: print(f"Epoch {epoch + 1}/{epochs}")

            indices = np.random.permutation(n_samples)
            print("Training epoch ", epoch + 1, "with batch size", batch_size)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            batch_n = 1
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Train within the current branch
                self._forward_propagation(X_batch)
                self._compute_loss(y_batch, batch_n)
                self._back_propagation(y_batch)
                self._update_weights()
                batch_n += 1

        if self.verbose: print("Finished training")

    def predict(self, X_test):
        """
        Makes predictions on a data set X_test.

        Input parameters:
            - X_test: Input data for which to make predictions of shape (n_samples, n_features).

        Returns:
            - predictions: Predicted values.
        """
        if self.weights is None or self.biases is None:
            raise RuntimeError("The model has not been trained yet. Call train() first.")

        # Ensure X_test is a 2D NumPy array
        X_test = np.asarray(X_test)
        
        expected_features = self.weights[0].shape[1]

        if X_test.ndim == 1:
            if X_test.shape[0] == expected_features:
                X_test = X_test.reshape(1, -1) 
            else:
                raise ValueError(
                    f"Input X_test is 1D with {X_test.shape[0]} elements. "
                    f"If it's a single sample, it should have {expected_features} features. "
                    "Otherwise, X_test should be 2D (samples, features)."
                )
        elif X_test.ndim != 2:
            raise ValueError(f"Input data X_test must be 2-dimensional (samples, features). Got {X_test.ndim} dimensions.")

        # Check if number of features in X_test matches model's input features
        if X_test.shape[1] != expected_features:
            raise ValueError(
                f"Input X_test has {X_test.shape[1]} features, "
                f"but the model was trained/initialized expecting {expected_features} features."
            )

        # Perform forward propagation for prediction 
        A_prev = X_test
        for layer_idx in range(self.num_layers):
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]
            
            Z_current = np.dot(A_prev, W.T) + b
            A_current = self._activation(layer_idx, Z_current) 
            
            A_prev = A_current 
        
        y_pred_raw = A_prev 

        if self.task == 'classification':
            predictions = np.argmax(y_pred_raw, axis=1)
        elif self.task == 'regression':
            predictions = y_pred_raw.squeeze() if y_pred_raw.shape[1] == 1 else y_pred_raw
            
        return predictions
        
    def _set_activation_functions(self, activation_functions):
        """
        Set the activation function of each hidden-layer, only allowing those predefined in ALLOWED_HIDDEN_ACTIVATION_FUNCTIONS.

        Input parameters:
            - activation_functions: list of activation functions. If len == 1, the same activation function will be used for 
                                    all layers. Otherwise, each layer will have the corresponding activation function.
        """

        # Check if the activation functions are implemented
        for act in activation_functions:
            if act not in self.ALLOWED_HIDDEN_ACTIVATION_FUNCTIONS:
                raise ValueError(f"Activation function {act} is not implemented yet for hidden-layers. Only 'ReLU' is allowed.")

        # Store the activation functions of the hidden layers
        if len(activation_functions) == 1: act_func = activation_functions * self.num_hidden_layers
        elif len(activation_functions) == self.num_hidden_layers: act_func = activation_functions
        else: raise ValueError(f"Number of activation functions ({len(activation_functions)}) does not match number of layers ({self.num_hidden_layers})")

        # Include the output activation function
        if self.task == 'classification': act_func.append('softmax')
        elif self.task == 'regression': act_func.append('linear')

        return act_func

    def _fit(self, X, y, batch_size):
        """
        Initializes the weight matrices and bias vectors according to the provided data and task.
        
        Input Parameters:
            - X : training data features
            - y: training target
            - batch_size: size of the mini-batch for training
        """
        # Some checks on the data:
        X = np.asarray(X) 
        if X.ndim < 2: raise ValueError(f"Input data X must be 2-dimensional (samples, features). Got {X.ndim} dimensions.")
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]: raise ValueError(f"X and y must have the same number of samples")

        # Initialize the weights and biases
        self._init_weights(input_features = X.shape[1], target = y)

        # Initialize the layers
        self._init_layers(batch_size, target = y)

        if self.verbose:      
            print("#################################################################")
            print("Initial weights and biases: ")
            print(f"Weights: {self.weights}")
            print(f"Biases: {self.biases}")
            print(f"Layers: {self.layers}")

    def _init_weights(self, input_features, target):
        """
        Initialize the weights (He initialization) of the Multi Layer Perceptron for all hidden layers.

        Input parameters:
            - input_features: Dimensionality(number of features) of the training data.
            - target: target variable containing the target for all training samples.
        """

        # Initialize the hidden-layer weights and biases
        weights = []
        biases = []
        for layer in range(self.num_hidden_layers):
            if layer == 0: weight_matrix = np.random.randn(self.layer_neurons[layer],input_features) * np.sqrt(2 / input_features)
            else: weight_matrix = np.random.randn(self.layer_neurons[layer],self.layer_neurons[layer-1]) * np.sqrt(2 /self.layer_neurons[layer-1])
            bias_vector = np.zeros(self.layer_neurons[layer])
            weights.append(weight_matrix)
            biases.append(bias_vector)

        # Initialize the output layer weights and biases
        self.output_neurons = len(np.unique(target)) if self.task == 'classification' else 1
        if self.num_hidden_layers == 0:
            num_neurons_in_prev_layer = input_features
        else:
            num_neurons_in_prev_layer = self.layer_neurons[-1]
        
        output_layer_weights = np.random.randn(self.output_neurons, num_neurons_in_prev_layer) * np.sqrt(2 / num_neurons_in_prev_layer)
        weights.append(output_layer_weights)
        biases.append(np.zeros(self.output_neurons))

        # Save the weights and biases
        self.weights = weights
        self.biases = biases

    def _init_layers(self, batch_size, target):
        """
        Initializes layer values to 0.

        Input parameters:
            - batch_size: size of the mini-batch for training
            - target: target variable containing the target for all training samples.
        """

        layers = []
        # Initialize Hidden Layers
        for layer in range(self.num_hidden_layers):
            layers.append(np.zeros(shape=(batch_size,self.layer_neurons[layer])))
        
        # Initialize Output Layer
        if self.task == 'classification': layers.append(np.zeros(shape=(batch_size,len(np.unique(target)))))
        elif self.task == 'regression': layers.append(np.zeros(shape=(batch_size,1)))
        self.layers = layers

    def _forward_propagation(self, X):
        """
        Performs a forward step through the network.
        
        Input parameters:
            - X: Input data of shape (n_samples, n_features).
        """
        self.X_batch_current = X  
        self.cache_Z = []

        A = X 

        for layer in range(self.num_layers): # 0 to num_layers-1
            W = self.weights[layer]
            b = self.biases[layer]
            
            # Z = A . W^T + b
            Z = np.dot(A, W.T) + b 
            self.cache_Z.append(Z)
            
            self.layers[layer] = self._activation(layer, Z)             
            A = self.layers[layer]

    def _activation(self, layer, Z):
        """
        Computes and returns the activation layer.

        Input parameters:
            - layer: Index of the current layer.
            - Z: Pre-activation values for the current layer.
        """
        if self.activation_functions[layer] == 'ReLU': return np.maximum(0, Z)
        elif self.activation_functions[layer] == 'linear': return Z
        elif self.activation_functions[layer] == 'softmax': 
            # We substract max(z) to all elements to avoid big exponentials. 
            # Softmax is shift invariant, so no problem with that.
            e_z = np.exp(Z - np.max(Z, axis=1, keepdims=True)) 
            return e_z / np.sum(e_z, axis=1, keepdims=True)

    def _compute_loss(self, y_true_batch, batch_n, return_only=False): 
        """
        Computes the loss function.
        
        Input parameters:
            - y_true_batch: True labels for the current batch.
            - batch_n: Index of the current batch.
            - return_only: If True, only returns the loss value without printing it.

        Returns:
            - loss: Computed loss value.
        """
        y_pred = self.layers[-1]

        if self.task == 'classification':
            epsilon = 1e-12
            num_samples = y_true_batch.shape[0]
            correct_class_probabilities = y_pred[np.arange(num_samples), y_true_batch.astype(int)]
            log_likelihood = np.log(correct_class_probabilities + epsilon)
            loss = -np.mean(log_likelihood)
        elif self.task == 'regression':
            y_true_reshaped = y_true_batch.reshape(-1, 1)
            loss = np.mean((y_pred - y_true_reshaped)**2)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        if not return_only:
            if self.verbose: print(f"Loss for batch {batch_n} = {loss}")
            self.loss = loss

        return loss 

    def _back_propagation(self, y_batch):
        """
        Back-propagation step to compute gradients of the loss function.
        Computes the gradient of the loss function with respect to all 
        parameters using the back-propagation algorithm.
        Stores computed gradients in self.dW and self.db.

        Note: All derivations can be found in the report.

        Input parameters:
            - y_batch: True labels for the current batch.
        """
        
        self.dW = [np.zeros_like(w) for w in self.weights]
        self.db = [np.zeros_like(b) for b in self.biases]
        m = y_batch.shape[0] 

        # Step 1: Compute gradients for the output layer
        if self.task == 'classification':
            # For Softmax + Cross-Entropy loss
            y_one_hot = np.zeros_like(self.layers[-1])
            y_one_hot[np.arange(m), y_batch] = 1
            dZ = 1/m * (self.layers[-1] - y_one_hot)
        
        elif self.task == 'regression':
            # For Linear output + Mean Squared Error loss:
            y_true_reshaped = y_batch.reshape(-1, 1)
            dZ = (2 / m) * (self.layers[-1] - y_true_reshaped)

        if self.num_hidden_layers > 0:
            A_prev = self.layers[-2] 
        else: 
            A_prev = self.X_batch_current 

        self.dW[-1] = np.dot(dZ.T, A_prev)
        self.db[-1] = np.sum(dZ, axis=0)

        # Step 2: Propagate gradients backwards through hidden layers
        for layer in range(self.num_layers - 2, -1, -1):

            W_next = self.weights[layer + 1]  
            dZ_next = dZ  

            dA_curr = np.dot(dZ_next, W_next)
            
            Z_curr = self.cache_Z[layer] 
            activation_fn_curr = self.activation_functions[layer]
            
            if activation_fn_curr == 'ReLU':
                d_activation = np.where(Z_curr > 0, 1, 0) 
            else:
                raise ValueError(f"Derivative for activation function {activation_fn_curr} not implemented.")

            dZ = dA_curr * d_activation 

            if layer == 0:
                A_prev_curr = self.X_batch_current
            else:
                A_prev_curr = self.layers[layer - 1] 
            
            self.dW[layer] = np.dot(dZ.T, A_prev_curr)
            self.db[layer] = np.sum(dZ, axis=0)

    def _update_weights(self):
        """ 
        Uses Gradient Descent to update the weights and biases of the MLP.
        Assumes self.learning_rate is set and self.dW, self.db contain gradients.
        """

        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * self.dW[i]
            self.biases[i] -= self.learning_rate * self.db[i]

    def gradient_check(self, X, y, epsilon=1e-5, tolerance=1e-7):
        """
        Performs numerical gradient checking for ALL weights and biases.
        Compares backpropagation gradients with numerical approximation.

        Input parameters:
            - X: Input data of shape (n_samples, n_features).
            - y: Target data of shape (n_samples, n_classes) for classification or (n_samples,) for regression.
            - epsilon: Small value for numerical gradient approximation.
            - tolerance: Tolerance for gradient check failure.
        Returns:
            - relative_diff: Relative difference between analytical and numerical gradients.
        """

        # 0. Store original parameters to restore them later and prevent state changes
        original_weights = [w.copy() for w in self.weights]
        original_biases = [b.copy() for b in self.biases]

        # 1. Compute analytical gradients using current parameters
        self._forward_propagation(X)
        self._back_propagation(y) 

        analytical_grads_flat = []
        for dW_layer in self.dW:
            analytical_grads_flat.extend(dW_layer.flatten().tolist())
        for db_layer in self.db:
            analytical_grads_flat.extend(db_layer.flatten().tolist())
        
        numerical_grads_flat = []

        print("Checking Weights...")
        # Iterate through each weight matrix
        for l_idx in range(len(self.weights)):
            W_layer_ref = self.weights[l_idx] 
            print(f"  Layer {l_idx} weights ({W_layer_ref.shape})...")
            # Iterate through each element of the weight matrix
            for i in range(W_layer_ref.shape[0]):
                for j in range(W_layer_ref.shape[1]):
                    original_val = W_layer_ref[i, j] 

                    # Calculate loss for W_ij + epsilon
                    W_layer_ref[i, j] = original_val + epsilon
                    self._forward_propagation(X) 
                    loss_plus = self._compute_loss(y, batch_n=1, return_only=True)

                    # Calculate loss for W_ij - epsilon
                    W_layer_ref[i, j] = original_val - epsilon
                    self._forward_propagation(X) 
                    loss_minus = self._compute_loss(y, batch_n=1, return_only=True)

                    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
                    numerical_grads_flat.append(numerical_grad)
                    
                    # Restore original value for this specific weight
                    W_layer_ref[i, j] = original_val 
        print("Checking Biases...")
        # Iterate through each bias vector
        for l_idx in range(len(self.biases)):
            b_layer_ref = self.biases[l_idx] # This is a reference
            print(f"  Layer {l_idx} biases ({b_layer_ref.shape})...")
            # Iterate through each element of the bias vector
            for i in range(b_layer_ref.shape[0]):
                original_val = b_layer_ref[i]

                # Calculate loss for b_i + epsilon
                b_layer_ref[i] = original_val + epsilon
                self._forward_propagation(X) 
                loss_plus = self._compute_loss(y, batch_n=1, return_only=True)

                # Calculate loss for b_i - epsilon
                b_layer_ref[i] = original_val - epsilon
                self._forward_propagation(X) 
                loss_minus = self._compute_loss(y, batch_n=1, return_only=True)
                
                numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
                numerical_grads_flat.append(numerical_grad)

                # Restore original value for this specific bias
                b_layer_ref[i] = original_val 

        # 3. Restore all original parameters from copies to ensure MLP state is unchanged
        self.weights = [w.copy() for w in original_weights] 
        self.biases = [b.copy() for b in original_biases]

        # 4. Compute and compare relative errors
        analytical_grads_flat = np.array(analytical_grads_flat)
        numerical_grads_flat = np.array(numerical_grads_flat)

        # Ensure dimensions match before comparison (debugging step)
        if analytical_grads_flat.shape != numerical_grads_flat.shape:
            print(f"WARNING: Shape mismatch! Analytical: {analytical_grads_flat.shape}, Numerical: {numerical_grads_flat.shape}")
        
        numerator = np.linalg.norm(analytical_grads_flat - numerical_grads_flat)
        denominator = np.linalg.norm(analytical_grads_flat) + np.linalg.norm(numerical_grads_flat)
        
        relative_diff = numerator / (denominator + 1e-12) # Add epsilon to avoid division by zero

        print(f"Overall gradient relative error: {relative_diff:.2e}")

        if relative_diff >= tolerance:
            raise AssertionError(f"Gradient check failed: Relative error {relative_diff:.2e} >= tolerance {tolerance:.2e}")
        
        print("Gradient check passed.")
        return relative_diff