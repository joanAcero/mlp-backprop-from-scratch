# MLP from Scratch with Backpropagation

This project implements a Multi-Layer Perceptron (MLP) neural network from scratch using only Python and NumPy. It focuses on the theoretical derivation, complexity analysis, and practical implementation of the backpropagation algorithm for training MLPs.

The MLP is capable of performing both K-class classification (using softmax output and cross-entropy loss) and unidimensional regression (using linear output and mean squared error loss). Hidden layers utilize the ReLU activation function.

A detailed theoretical report accompanying this project can be found in `report/report.pdf`.

## Features

*   **MLP Implementation from Scratch:** Core neural network components built using only NumPy.
*   **Backpropagation Algorithm:** Analytically derived and implemented for efficient gradient computation.
*   **Versatile Tasks:**
    *   K-class Classification (Softmax, Cross-Entropy Loss)
    *   Unidimensional Regression (Linear Output, Mean Squared Error Loss)
*   **Configurable Architecture:** Define the number of hidden layers and neurons per layer.
*   **Activation Functions:** ReLU for hidden layers, Softmax/Linear for output layers.
*   **Numerical Gradient Checking:** Implemented to verify the correctness of the backpropagation algorithm.
*   **Mini-Batch Gradient Descent:** Used for training the network.
*   **Detailed Report:** Comprehensive theoretical derivation, complexity analysis, and implementation details.

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

