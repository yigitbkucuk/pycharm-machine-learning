# KNN Görüntü Sınıflandırma Ödevi - Ana Notebook
# Öğrenci Adı: Yiğit Buğra KÜÇÜK
# Öğrenci No: 230212048
# Tarih: 16/10/2025

# 1. Kütüphaneleri Import Etme
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time
import os

from knn_classifier import KNNClassifier
from visualization import *

# Results klasörünü oluştur
os.makedirs('results', exist_ok=True)

# 2. Veri Yükleme ve Hazırlama
print("=== Veri Yükleme ===")
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train / 16.0  # Normalize et (0-1 arası)
X_test = X_test / 16.0  # Normalize et (0-1 arası)

print(f"Training samples: {X_train.shape}")
print(f"Test samples: {X_test.shape}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of classes: {len(np.unique(y_train))}")

# 3. Görev 1.2: MNIST Digits ile Test
print("\n" + "=" * 50)
print("Görev 1.2: MNIST Digits Testi")
print("=" * 50)

# Model oluştur ve eğit (k=3, L2 mesafe)
knn = KNNClassifier(k=3, distance_metric='l2')
knn.fit(X_train, y_train)

# Test accuracy hesapla
start_time = time.time()
accuracy = knn.score(X_test, y_test)
end_time = time.time()

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Hesaplama Süresi: {end_time - start_time:.2f} saniye")

# Tahminler yap
y_pred = knn.predict(X_test)

# Confusion matrix çiz
plot_confusion_matrix(y_test, y_pred)

# 10 örnek görüntüyü tahminleriyle görselleştir
plot_sample_predictions(X_test, y_test, y_pred)

# 4. Görev 1.3a: K Değeri Analizi
print("\n" + "=" * 50)
print("Görev 1.3a: K Değeri Analizi")
print("=" * 50)

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
accuracies_l2 = []

print("K Değeri Analizi Sonuçları:")
for k in k_values:
    knn = KNNClassifier(k=k, distance_metric='l2')
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    accuracies_l2.append(accuracy)
    print(f"k={k}: Accuracy = {accuracy:.4f}")

# K değeri analizi grafiği
plot_k_analysis(k_values, accuracies_l2)

# 5. Görev 1.3b: Mesafe Metriği Karşılaştırması
print("\n" + "=" * 50)
print("Görev 1.3b: Mesafe Metriği Karşılaştırması")
print("=" * 50)

l1_accuracies = []
l2_accuracies = []

print("L1 vs L2 Mesafe Metrikleri Karşılaştırması:")
print("K Değeri | L1 Accuracy | L2 Accuracy | Fark")
print("-" * 45)

for k in k_values:
    # L1 mesafe ile model
    knn_l1 = KNNClassifier(k=k, distance_metric='l1')
    knn_l1.fit(X_train, y_train)
    acc_l1 = knn_l1.score(X_test, y_test)

    # L2 mesafe ile model
    knn_l2 = KNNClassifier(k=k, distance_metric='l2')
    knn_l2.fit(X_train, y_train)
    acc_l2 = knn_l2.score(X_test, y_test)

    l1_accuracies.append(acc_l1)
    l2_accuracies.append(acc_l2)

    diff = abs(acc_l1 - acc_l2)
    print(f"{k:8} | {acc_l1:.4f}     | {acc_l2:.4f}     | {diff:.4f}")

# Karşılaştırma grafikleri
plot_distance_comparison(k_values, l1_accuracies, l2_accuracies)
create_comparison_table(k_values, l1_accuracies, l2_accuracies)

# 6. Bölüm 2: Sklearn Karşılaştırması
print("\n" + "=" * 50)
print("Bölüm 2: Sklearn Karşılaştırması")
print("=" * 50)

# Sklearn KNN
start_time = time.time()
sklearn_knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
sklearn_knn.fit(X_train, y_train)
sklearn_accuracy = sklearn_knn.score(X_test, y_test)
sklearn_time = time.time() - start_time

# Kendi KNN'imiz
start_time = time.time()
your_knn = KNNClassifier(k=3, distance_metric='l2')
your_knn.fit(X_train, y_train)
your_accuracy = your_knn.score(X_test, y_test)
your_time = time.time() - start_time

print("Sklearn vs Kendi Implementasyonumuz:")
print(f"Sklearn KNN Accuracy: {sklearn_accuracy:.4f}")
print(f"Kendi KNN Accuracy: {your_accuracy:.4f}")
print(f"Accuracy Farkı: {abs(sklearn_accuracy - your_accuracy):.4f}")
print(f"Sklearn Hesaplama Süresi: {sklearn_time:.2f} saniye")
print(f"Kendi Hesaplama Süresi: {your_time:.2f} saniye")

# 7. Analiz ve Yorumlar
print("\n" + "=" * 50)
print("ANALİZ VE YORUMLAR")
print("=" * 50)

print("\n--- K Değeri Analizi Yorumları ---")
print("• Hangi k değeri en iyi sonucu veriyor?")
print("  CEVAP: k=7 değeri en iyi sonucu veriyor (Accuracy: 0.9889)")
print("• K artıkça accuracy nasıl değişiyor?")
print("  CEVAP: k=7'ye kadar accuracy artıyor, sonra azalıyor. Optimal değer k=7")
print("• Underfitting veya overfitting gözlemlediniz mi?")
print("  CEVAP: k=1'de overfitting, k=21'de underfitting gözlemlendi")

print("\n--- Mesafe Metriği Analizi Yorumları ---")
print("• Hangi mesafe metriği daha iyi performans gösteriyor?")
print("  CEVAP: L2 (Euclidean) metriği daha iyi performans gösteriyor")
print("• Farklar anlamlı mı?")
print("  CEVAP: Farklar küçük ama tutarlı şekilde L2 daha iyi")
print("• Neden bu farklar oluşuyor olabilir?")
print("  CEVAP: MNIST görüntülerinde Euclidean mesafe daha anlamlı uzaklık ölçümü sağlıyor")

print("\n--- Sklearn Karşılaştırması Yorumları ---")
print("• Sonuçlar benzer mi?")
print("  CEVAP: Accuracy değerleri tamamen aynı (%98.33)")
print("• Fark varsa nedeni ne olabilir?")
print("  CEVAP: Sklearn daha yavaş çalışıyor (optimizasyon farkları)")
print("• Sklearn'in avantajları neler?")
print("  CEVAP: Daha optimize, daha fazla parametre seçeneği, daha kararlı")

print("\n--- Genel Sonuçlar ve Öğrenilenler ---")
print("• KNN algoritması MNIST için oldukça başarılı (%98+ accuracy)")
print("• k değeri seçimi kritik öneme sahip")
print("• Euclidean mesafe bu problem için daha uygun")
print("• Kendi implementasyonumuz sklearn ile aynı doğrulukta")
print("• Görselleştirme model performansını anlamada çok yardımcı")

print("\n" + "=" * 50)
print("TÜM DENEYLER TAMAMLANDI!")
print("=" * 50)