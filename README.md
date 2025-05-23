# MLP from Scratch with Backpropagation

This project implements a Multi-Layer Perceptron (MLP) neural network from scratch using only Python and NumPy. It focuses on the theoretical derivation, complexity analysis, and practical implementation of the backpropagation algorithm for training MLPs.

---

📘 **[Read the Full Theoretical Report (PDF)](https://github.com/joanAcero/mlp-from-scratch-backpropagation/blob/main/report/Report.pdf)**  
Includes all required background, derivations, and the math behind the implementation.

---

The MLP is capable of performing both K-class classification (using softmax output and cross-entropy loss) and unidimensional regression (using linear output and mean squared error loss). Hidden layers utilize the ReLU activation function.


## Features

*   **Detailed Report:** Comprehensive theoretical derivation, complexity analysis, and implementation details.
*   **MLP Implementation from Scratch:** Core neural network components built using only NumPy.
*   **Configurable Architecture:** Define the number of hidden layers and neurons per layer.
*   **Activation Functions:** ReLU for hidden layers, Softmax/Linear for output layers.
*   **Versatile Tasks:**
    *   K-class Classification 
    *   Regression 
*   **Backpropagation Algorithm:** Analytically derived and implemented for efficient gradient computation.
*   **Numerical Gradient Checking:** Implemented to verify the correctness of the backpropagation algorithm.

## Getting Started

### Prerequisites

*   NumPy


### Usage

The `MLP.py` file in the `mlp/` directory contains the `MLP` class. You can import and use this class in your own Python scripts.

An example of how to use the MLP for both classification and regression tasks can be found in `examples/exampleusage.py`.


### Contributing
Contributions, issues, and feature requests are welcome. Please feel free to open an issue or submit a pull request.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

