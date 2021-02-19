import csv
from collections import namedtuple
import math
import random
import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def initialize_mnist(neural_network, training_count=1000):
    with open("mnist_train.csv", "rb") as f:
        data_matrix = np.loadtxt(f, delimiter=",", skiprows=1)
    data_labels = data_matrix[:,0].astype(int)
    data_values = data_matrix[:,1:]
    data_values = np.where(data_values > 180, 1, 0)
    pool = list(zip(data_values, data_labels))
    random.shuffle(pool)
    for value, label in pool[:training_count]:
        neural_network.back_propagate(value, label)

class OCRNeuralNetwork:
    LEARNING_RATE = 0.2
    NEURAL_NETWORK_FILE_PATH = "neural_network.json"
    def __init__(
            self,
            num_input_nodes,
            num_hidden_nodes,
            num_output_nodes,
            load_from_file=None
    ):
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = num_output_nodes
        self.__sigmoid = np.vectorize(self._sigmoid_scalar)
        self.__sigmoid_prime = np.vectorize(self._sigmoid_prime_scalar)
        if load_from_file is None:
            self.theta1 = self._initialize_random_weights(num_input_nodes, num_hidden_nodes)
            self.theta2 = self._initialize_random_weights(num_hidden_nodes, num_output_nodes)
            self.input_layer_bias = np.random.rand(num_hidden_nodes) * 0.12 - 0.06
            self.hidden_layer_bias = np.random.rand(num_output_nodes) * 0.12 - 0.06
        else:
            self.load(load_from_file)

    def _initialize_random_weights(self, size_in, size_out):
        """
        Creates a matrix with `size_out` rows and `size_in` columns.
        Values will be randomized between -0.06 and 0.06.
        """
        return np.random.rand(size_in, size_out) * 0.12 - 0.06

    def sigmoid(self, z):
        return self.__sigmoid(np.clip(z, -100, 100))

    def _sigmoid_scalar(self, z):
        """Activation function."""
        return 1 / (1 + math.e ** -z)

    def sigmoid_prime(self, z):
        return self.__sigmoid_prime(np.clip(z, -100, 100))

    def _sigmoid_prime_scalar(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def initialize(self):
        with open("simple_train.csv", "rb") as f:
            data_matrix = np.loadtxt(f, delimiter=",", skiprows=1)
        data_labels = data_matrix[:1000,0]
        data_matrix = data_matrix[:1000,1:]
        # data_matrix = np.where(data_matrix > 160, 1, 0)
        data_with_labels = list(zip(data_matrix, data_labels))
        for data, label in random.choices(data_with_labels, k=1000):
            self.back_propagate(data, int(label))

    def forward_propagate(self, input_vals):
        input_vals = np.array(input_vals)
        y1 = np.dot(input_vals, self.theta1)
        y1 += self.input_layer_bias
        y1 = self.sigmoid(y1)

        y2 = np.dot(y1, self.theta2)
        y2 += self.hidden_layer_bias
        y2 = self.sigmoid(y2)
        return y2

    def predict(self, test):
        output_node_vals = self.forward_propagate(test)
        return output_node_vals

    def back_propagate(self, input_data, data_label):
        # Step 1. Forward propagate, saving the intermediate values
        # that we'll need for the backprop partial derivative formula later.

        # Save off this pre-activation value. We need it later.
        hidden_layer_pre_activations = (
            np.dot(input_data, self.theta1)
            + self.input_layer_bias
        )
        hidden_layer_activations = self.sigmoid(hidden_layer_pre_activations)

        output_layer_pre_activations = (
            np.dot(hidden_layer_activations, self.theta2)
            + self.hidden_layer_bias
        )
        output_layer_activations = self.sigmoid(output_layer_pre_activations)
        self.output_layer_activations = output_layer_activations

        # Step 2. Back propagate.
        target_values = np.zeros(self.num_output_nodes)
        target_values[data_label] = 1


        # 1 x num_output_nodes
        errors_of_output_layer = output_layer_activations - target_values
        self.errors = errors_of_output_layer

        # num_output_nodes x num_hidden_nodes
        # same dimensions as weights
        rate_of_change_of_error_with_respect_to_final_weights = np.dot(
            (
                errors_of_output_layer
                * self.sigmoid_prime(output_layer_pre_activations)
            ).reshape(-1, 1),
            hidden_layer_activations.reshape(1, -1)
        ).T
        self.rate_of_change_of_error_with_respect_to_final_weights = (
            rate_of_change_of_error_with_respect_to_final_weights
        )

        # 1 x num_hidden_nodes
        errors_of_hidden_layer = np.dot(
            errors_of_output_layer
            * self.sigmoid_prime(output_layer_pre_activations),
            self.theta2.T
        )
        self.errors_of_hidden_layer = errors_of_hidden_layer
        # num_hidden_nodes x num_input_nodes
        # same dimensions as weights
        rate_of_change_of_error_with_respect_to_first_weights = (
            (
                errors_of_hidden_layer  # 1 x num_hidden_nodes
                * self.sigmoid_prime(hidden_layer_pre_activations)  # 1 x num_hidden_nodes
            ).reshape(-1, 1)  # num_hidden_nodes x 1
            * input_data.reshape(1, -1)  # 1 x num_input_nodes
        ).T

        self.theta2 -= (
            self.LEARNING_RATE
            * rate_of_change_of_error_with_respect_to_final_weights
        )
        self.hidden_layer_bias -= errors_of_output_layer * self.LEARNING_RATE
        self.theta1 -= (
            self.LEARNING_RATE
            * rate_of_change_of_error_with_respect_to_first_weights
        )
        self.input_layer_bias -= errors_of_hidden_layer * self.LEARNING_RATE

    def save(self, filepath=None):
        """
        We need to work with Numpy "array" types, but the `json` library
        that we use to serialize/deserialize doesn't know about Numpy types.
        So, we serialize things as regular python types, like lists, and then
        deserialize them the same way, and then convert them back to Numpy types.
        """
        json_neural_network = {
            "theta1": self.theta1.tolist(),
            "theta2": self.theta2.tolist(),
            "bias1": self.input_layer_bias.tolist(),
            "bias2": self.hidden_layer_bias.tolist(),
        }
        filepath = filepath or self.NEURAL_NETWORK_FILE_PATH
        with open(filepath, "w") as f:
            json.dump(json_neural_network, f)

    def load(self, filepath):
        """
        We need to work with Numpy "array" types, but the `json` library
        that we use to serialize/deserialize doesn't know about Numpy types.
        So, we serialize things as regular python types, like lists, and then
        deserialize them the same way, and then convert them back to Numpy types.
        """
        if not os.path.isfile(filepath):
            return
        with open(filepath) as f:
            neural_network = json.load(f)
        self.theta1 = np.array(neural_network["theta1"])
        self.theta2 = np.array(neural_network["theta2"])
        self.input_layer_bias = np.array(neural_network["bias1"])
        self.hidden_layer_bias = np.array(neural_network["bias2"])
