import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, k=3, distance_metric='l2'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances(self, X):
        n_test = X.shape[0]
        n_train = self.X_train.shape[0]
        distances = np.zeros((n_test, n_train))

        if self.distance_metric == 'l1':
            for i in range(n_test):
                distances[i, :] = np.sum(np.abs(self.X_train - X[i]), axis=1)
        elif self.distance_metric == 'l2':
            for i in range(n_test):
                distances[i, :] = np.sqrt(np.sum((self.X_train - X[i]) ** 2, axis=1))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def predict(self, X):
        distances = self.compute_distances(X)
        predictions = []

        for i in range(distances.shape[0]):
            k_indices = np.argsort(distances[i])[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])

        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy