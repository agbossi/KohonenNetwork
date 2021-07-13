import math
import random
import sys

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.utils import shuffle


class KohonenNetwork:
    def __init__(self, network_size, input_size, network_type):
        self.network = [[KohonenNeuron(input_size, j, i) for j in range(network_size)] for i in range(network_size)]
        self.input_size = input_size
        self.network_size = network_size
        self.type = network_type
        self.training_set = None
        self.labels = {}

    # si hay datos no numericos que identifican, se asume que es la primera columna
    def initialize(self, training_set, labels_map):
        self.labels = labels_map
        self.training_set = training_set
        for i in range(self.network_size):
            for j in range(self.network_size):
                self.network[i][j].weights = np.array(self.training_set[random.randint(0, len(self.training_set)-1)])

    def train(self, iterations, learning_rate_function, radius_function):
        for i in range(iterations):
            self.clear_marks()
            self.training_set = shuffle(self.training_set)
            for training_example in self.training_set:
                winner_neuron = self.get_representative(training_example)
                vicinity = self.get_vicinity(winner_neuron, radius_function.get_next_radius())  # por ahora radio es cte
                self.update(vicinity, training_example, learning_rate_function.get_next_learning_rate())

    def get_representative(self, input_example):
        min_distance = sys.maxsize
        winner = None
        for i in range(self.network_size):
            for j in range(self.network_size):
                curr_distance = self.network[i][j].get_euclidean_distance(input_example)
                if curr_distance < min_distance:
                    min_distance = curr_distance
                    winner = self.network[i][j]
        winner.times_chosen += 1
        winner.represents.append(input_example)
        return winner

    def clear_marks(self):
        for i in range(self.network_size):
            for j in range(self.network_size):
                self.network[i][j].times_chosen = 0
                self.network[i][j].represents.clear()

    def update(self, vicinity, training_example, learning_rate):
        for neuron in vicinity:
            delta = learning_rate * np.subtract(training_example, neuron.weights) * (1 / neuron.distance_from_winner)
            neuron.weights = np.add(neuron.weights, delta)
            neuron.distance_from_winner = sys.maxsize

    def get_vicinity(self, winner_neuron, radius):
        winner_neuron.distance_from_winner = 1
        vicinity = [winner_neuron]
        for i in range(self.network_size):
            for j in range(self.network_size):
                if i != winner_neuron.y_coord or j != winner_neuron.x_coord:
                    self.network[i][j].distance_from_winner = winner_neuron.get_manhattan_grid_distance(self.network[i][j])
                    if self.network[i][j].distance_from_winner < radius:
                        vicinity.append(self.network[i][j])
        return vicinity

    def build_U_matrix(self):
        feature_map = [[self.get_average_neighbour_distance(self.network[i][j]) for j in range(self.network_size)] for i in range(self.network_size)]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(' ', ['white', 'grey', 'black'])
        plt.imshow(feature_map, cmap=cmap)
        plt.title("Matriz U")
        plt.colorbar()
        plt.axis('off')
        plt.show()

    def get_label(self, label_info):
        for key in self.labels:
            if np.array_equal(self.labels[key], label_info):
                return key
        return None
# definitivamente hay un problema con el represents, agrega otra cosa, o hay una modificacion de dataset
# o no se esta limpiando el represents

    def get_average_neighbour_distance(self, neuron):
        x_coord = neuron.x_coord
        y_coord = neuron.y_coord
        distances = 0
        distances_sum = 0

        # upper distance
        if y_coord-1 > 0:
            distances_sum += self.network[y_coord-1][x_coord].get_euclidean_distance(neuron.weights)
            distances += 1
        # lower distance
        if y_coord+1 < self.network_size:
            distances_sum += self.network[y_coord+1][x_coord].get_euclidean_distance(neuron.weights)
            distances += 1
        # get left distance
        if x_coord-1 > 0:
            distances_sum += self.network[y_coord][x_coord-1].get_euclidean_distance(neuron.weights)
            distances += 1
        # get right distance
        if x_coord+1 < self.network_size:
            distances_sum += self.network[y_coord][x_coord+1].get_euclidean_distance(neuron.weights)
            distances += 1

        if self.type == "hexagonal":
            # get right upper side
            if x_coord+1 < self.network_size and y_coord-1 > 0:
                distances_sum += self.network[y_coord-1][x_coord+1].get_euclidean_distance(neuron.weights)
                distances += 1
            # get right lower side
            if x_coord+1 < self.network_size and y_coord+1 < self.network_size:
                distances_sum += self.network[y_coord+1][x_coord+1].get_euclidean_distance(neuron.weights)
                distances += 1
        neuron.avg_distance = distances_sum / distances
        return neuron.avg_distance

    def plot_network_features(self, variable_names, standardizator=None):
        for k in range(self.input_size):
            feature_map = [[self.network[i][j].weights[k] for j in range(self.network_size)] for i in range(self.network_size)]
            plt.title(variable_names[k])
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(' ', ['blue', 'yellow', 'red'])
            plt.imshow(feature_map, cmap=cmap)
            plt.colorbar()
            plt.axis('off')
            plt.show()

    def plot_clusters(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        A = []#[x for x in range(self.network_size)]
        B = []#[y for y in range(self.network_size)]  # esto tiene que tener la coordenada de los representantes

        for i in range(self.network_size):
            for j in range(self.network_size):
                k = 0
                for label in self.network[i][j].represents:
                    A.append(j)
                    B.append(i)
                    ax.annotate('%s' % self.get_label(label), xy=(j, i), xytext=(10, 20-10*k), textcoords='offset points')  # xy es la label?
                    k += 1

        plt.xlim(0, self.network_size-1)
        plt.ylim(0, self.network_size-1)
        plt.gca().invert_yaxis()
        plt.scatter(A, B)
        plt.grid()
        plt.show()

    def plot_network_hit_map(self):
        hit_map = [[self.network[i][j].times_chosen for j in range(self.network_size)] for i in range(self.network_size)]
        plt.title("cantidad de registros que van a cada nodo")
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(' ', ['blue', 'yellow', 'red'])
        plt.imshow(hit_map, cmap=cmap)
        plt.colorbar()
        plt.axis('off')
        plt.show()


class KohonenNeuron:
    def __init__(self, weights_amount, x_coord, y_coord):
        self.weights = np.zeros(weights_amount)
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.avg_distance = sys.maxsize
        self.times_chosen = 0
        self.distance_from_winner = sys.maxsize
        self.represents = []

    def get_euclidean_distance(self, coord):
        return np.linalg.norm(self.weights - coord)

    def get_manhattan_grid_distance(self, neuron):
        d = abs(self.x_coord - neuron.x_coord) + abs(self.y_coord - neuron.y_coord)
        return d

    def get_euclidean_grid_distance(self, neuron):
        d = math.sqrt(((self.x_coord - neuron.x_coord) ** 2) + ((self.y_coord - neuron.y_coord) ** 2))
        return d
