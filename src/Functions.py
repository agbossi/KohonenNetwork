import numpy as np

class LearningRateFunction:
    def __init__(self):
        self.learning_rate = 1
        self.count = 0

    def get_next_learning_rate(self):
        self.count += 1
        if self.learning_rate / (self.count / 5) <= 1:
            self.learning_rate = self.learning_rate / (self.count / 5)
        else:
            self.learning_rate = 1
        self.learning_rate
        return self.learning_rate


class RadiusFunction:
    def __init__(self, network_size, sorting_iterations_number):
        self.init_radius = network_size / 2
        self.count = 0
        self.sorting_iterations_number = sorting_iterations_number

    def get_next_radius(self):
        self.count += 1
        return self.init_radius * (self.sorting_iterations_number - self.count) / self.sorting_iterations_number

def Standarizer(data_set):
    #Get means per column
    col_means = np.sum(data_set, axis=0)
    col_means /= len(data_set)
    #Get std per column
    col_stds = np.zeros(len(data_set[0]))
    for i in range(len(data_set)):
        for j in range(len(data_set[0])):
            col_stds[j] += (data_set[i][j] - col_means[j])**2
    col_stds /= len(data_set)-1
    col_stds = np.sqrt(col_stds)

    for i in range(len(data_set)):
        for j in range(len(data_set[0])):
            data_set[i][j] = (data_set[i][j] - col_means[j])/col_stds[j]
    return data_set