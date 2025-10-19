import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time

from knn_classifier import KNNClassifier
from visualization import (
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_k_analysis,
    plot_distance_comparison,
    create_comparison_table
)


def load_data():
    """
    MNIST Digits veri setini yükler ve train/test'e böler
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train / 16.0
    X_test = X_test / 16.0
    return X_train, X_test, y_train, y_test


def test_mnist_basic():
    """
    Görev 1.2: MNIST üzerinde temel KNN testi
    """
    print("=" * 50)
    print("Görev 1.2: MNIST Digits Testi")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()

    knn = KNNClassifier(k=3, distance_metric='l2')
    knn.fit(X_train, y_train)

    start_time = time.time()
    accuracy = knn.score(X_test, y_test)
    end_time = time.time()

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Hesaplama Süresi: {end_time - start_time:.2f} saniye")

    y_pred = knn.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_sample_predictions(X_test, y_test, y_pred)

    return X_train, X_test, y_train, y_test, knn


def analyze_k_values():
    """
    Görev 1.3a: Farklı k değerlerinin etkisini analiz eder
    """
    print("\n" + "=" * 50)
    print("Görev 1.3a: K Değeri Analizi")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    accuracies = []

    for k in k_values:
        knn = KNNClassifier(k=k, distance_metric='l2')
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        accuracies.append(accuracy)
        print(f"k={k}: Accuracy = {accuracy:.4f}")

    plot_k_analysis(k_values, accuracies)
    return k_values, accuracies


def compare_distance_metrics():
    """
    Görev 1.3b: L1 ve L2 mesafe metriklerini karşılaştırır
    """
    print("\n" + "=" * 50)
    print("Görev 1.3b: Mesafe Metriği Karşılaştırması")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    l1_accuracies = []
    l2_accuracies = []

    print("K Değeri | L1 Accuracy | L2 Accuracy | Fark")
    print("-" * 45)

    for k in k_values:
        knn_l1 = KNNClassifier(k=k, distance_metric='l1')
        knn_l2 = KNNClassifier(k=k, distance_metric='l2')

        knn_l1.fit(X_train, y_train)
        knn_l2.fit(X_train, y_train)

        acc_l1 = knn_l1.score(X_test, y_test)
        acc_l2 = knn_l2.score(X_test, y_test)

        l1_accuracies.append(acc_l1)
        l2_accuracies.append(acc_l2)

        diff = abs(acc_l1 - acc_l2)
        print(f"{k:8} | {acc_l1:.4f}     | {acc_l2:.4f}     | {diff:.4f}")

    plot_distance_comparison(k_values, l1_accuracies, l2_accuracies)
    create_comparison_table(k_values, l1_accuracies, l2_accuracies)

    return k_values, l1_accuracies, l2_accuracies


def compare_with_sklearn():
    """
    Bölüm 2: Sklearn KNN ile karşılaştırma
    """
    print("\n" + "=" * 50)
    print("Bölüm 2: Sklearn Karşılaştırması")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()

    start_time = time.time()
    sklearn_knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    sklearn_knn.fit(X_train, y_train)
    sklearn_accuracy = sklearn_knn.score(X_test, y_test)
    sklearn_time = time.time() - start_time

    start_time = time.time()
    your_knn = KNNClassifier(k=3, distance_metric='l2')
    your_knn.fit(X_train, y_train)
    your_accuracy = your_knn.score(X_test, y_test)
    your_time = time.time() - start_time

    print(f"Sklearn KNN Accuracy: {sklearn_accuracy:.4f}")
    print(f"Kendi KNN Accuracy: {your_accuracy:.4f}")
    print(f"Fark: {abs(sklearn_accuracy - your_accuracy):.4f}")
    print(f"Sklearn Süre: {sklearn_time:.2f} saniye")
    print(f"Kendi Süre: {your_time:.2f} saniye")


if __name__ == "__main__":
    test_mnist_basic()
    analyze_k_values()
    compare_distance_metrics()
    compare_with_sklearn()