from Data_set_obtainer import DataSetObtainer
from Functions import RadiusFunction, LearningRateFunction
from Kohonen_network import KohonenNetwork
import numpy as np

#data_set = ["primero", "segundo", "tercero", "cuarto", "quinto", "sexto", "septimo", "octavo", "noveno", "decimo", "onceavo", "doceavo", "treceavo", "catorceavo", "quinceavo"]
#ds = np.array([[0.8, 0.8, 0.8], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.9, 0.9, 0.9], [0.15, 0.15, 0.15], [0.25, 0.25, 0.25], [0.85, 0.85, 0.85], [0.95, 0.95, 0.95], [0.12, 0.12, 0.12], [0.82, 0.82, 0.82], [0.72, 0.72, 0.72], [0.07, 0.07, 0.07], [0.13, 0.13, 0.13], [0.83, 0.83, 0.83], [0.97, 0.97, 0.97]])
#labels_map = {"primero": [0.8, 0.8, 0.8], "segundo": [0.1, 0.1, 0.1],
#              "tercero": [0.2, 0.2, 0.2], "cuarto": [0.9, 0.9, 0.9],
#              "quinto": [0.15, 0.15, 0.15], "sexto": [0.25, 0.25, 0.25],
#              "septimo": [0.85, 0.85, 0.85], "octavo": [0.95, 0.95, 0.95],
#              "noveno": [0.12, 0.12, 0.12], "decimo": [0.82, 0.82, 0.82],
#              "onceavo": [0.72, 0.72, 0.72], "doceavo": [0.07, 0.07, 0.07],
#              "treceavo": [0.13, 0.13, 0.13], "catorceavo": [0.83, 0.83, 0.83],
#              "quinceavo": [0.97, 0.97, 0.97]}

ds_obtainer = DataSetObtainer()
# devuelve etiquetas asociadas al dato si las hay y los datos numericos en numpy
dataset, labels_map = ds_obtainer.get_file('../europe.csv', "csv", True, True)

#data_set = np.array(df[1:]) # saco la primera fila con los indices
#x = data_set[:,1:] # saco la primera columna con los labels

# Standardizing the features

#labels_map = {}
#k = 0
#for row in data_set:
#    labels_map[row] = x[k] # reemplazar el row por row[0] para el posta
#    k += 1

network_size = int(len(dataset) / 5)  # 3 x 3
iterations = 5 * network_size
sorting_iterations = 900
radius = RadiusFunction(network_size, sorting_iterations)
learning_rate = LearningRateFunction()
input_size = 3

network = KohonenNetwork(network_size, input_size, "rectangular")
network.initialize(dataset, labels_map)
network.train(iterations, learning_rate, radius)
network.plot_clusters()
network.build_U_matrix()
network.plot_network_hit_map()
#network.plot_network_features(["Area", "GDP", "Inflation", "Life expect", "Military", "Pop growth", "Unemployment"], standardizator)
