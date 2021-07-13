import pandas as pd
from sklearn.preprocessing import StandardScaler
from Functions import Standarizer

class DataSetObtainer:
    def __init__(self):
        self.path = None

    def get_file(self, path, extension, has_header=True, has_label=False):
        self.path = path
        labels = []

        if extension == "xlsx":
            if has_header and ~has_label:
                data = pd.read_excel(self.path).iloc[1:, :]
            if has_header and has_label:
                data = pd.read_excel(self.path).iloc[1:, :]
                labels = data.iloc[:, 0]
                data = data.iloc[:, 1:]
            if ~has_header and has_label:
                data = pd.read_excel(self.path)
                labels = data.iloc[:, 0]
                data = data.iloc[:, 1:]
        elif extension == "csv":
            if has_header and not has_label:
                data = pd.read_csv(self.path).iloc[1:, :].to_numpy()
            if has_header and has_label:
                data = pd.read_csv(self.path).iloc[1:, :]
                labels = data.iloc[:, 0]
                data = data.iloc[:, 1:].to_numpy()
            if not has_header and has_label:
                data = pd.read_csv(self.path)
                labels = data.iloc[:, 0]
                data = data.iloc[:, 1:].to_numpy()
        else:
            data = None

        x = Standarizer(data)

        labels_map = {}
        k = 0
        for label in labels:
            labels_map[label] = x[k]
            k += 1
        return x, labels_map
