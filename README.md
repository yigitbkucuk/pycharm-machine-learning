# KNN Görüntü Sınıflandırma Ödevi

## Proje Açıklaması
Bu proje, K-En Yakın Komşuluk (KNN) algoritmasının sıfırdan implementasyonunu ve MNIST el yazısı rakamları veri seti üzerinde sınıflandırma performans analizini içermektedir. Çalışma, farklı parametrelerin (k değeri, mesafe metrikleri) model performansı üzerindeki etkisini detaylı şekilde incelemektedir.

## Proje Hedefleri
- KNN algoritmasının temel prensiplerinin anlaşılması ve implementasyonu
- Farklı k değerlerinin sınıflandırma performansı üzerindeki etkisinin analizi
- L1 (Manhattan) ve L2 (Euclidean) mesafe metriklerinin karşılaştırılması
- Scikit-learn kütüphanesi ile öz implementasyonun karşılaştırılması
- Görselleştirme teknikleri ile model performansının analiz edilmesi

## Gereksinimler
- **Python**: 3.8 veya üzeri
- **Gerekli Kütüphaneler**:
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Seaborn

## Kurulum

### 1. Gerekli Paketleri Yükleyin
```
pip install numpy scikit-learn matplotlib seaborn
```


### 2. Projeyi Çalıştırın
```
python main_notebook.py
```


## Proje Yapısı
Odev_Knn/
- ├── knn_classifier.py # KNN sınıfı implementasyonu
- ├── experiments.py # Deneyler ve testler
- ├── visualization.py # Görselleştirme fonksiyonları
- ├── main_notebook.py # Ana çalıştırma dosyası
- ├── KNN_Rapor.pdf # Proje raporu
- ├── README.md # Proje dokümantasyonu
- └── results/ # Sonuç grafikleri
- ├── confusion_matrix.png # Karışıklık matrisi
- ├── sample_predictions.png # Örnek tahmin görselleştirmeleri
- ├── k_value_analysis.png # K değeri analizi
- ├── comparison_table.png # Karşılaştırma tablosu
- └── distance_comparison.png # Mesafe metriği karşılaştırması



## Kullanım

### Temel Test
```
python experiments.py
```


### Tüm Analizler
```
python main_notebook.py
```


### Bireysel Fonksiyonlar
```
from knn_classifier import KNNClassifier
from visualization import plot_confusion_matrix
```
# Model oluştur ve eğit
```
knn = KNNClassifier(k=3, distance_metric='l2')
knn.fit(X_train, y_train)
```
# Tahmin yap ve görselleştir
```
predictions = knn.predict(X_test)
plot_confusion_matrix(y_test, predictions)
```

### Gerçekleştirilen Analizler
#### 1. Temel KNN Testi
- Model eğitimi: k=3, L2 mesafe metriği

- Test accuracy hesaplama

- Karışıklık matrisi oluşturma

- Örnek tahmin görselleştirmeleri

#### 2. K Değeri Analizi
- k = [1, 3, 5, 7, 9, 11, 15, 21] değerlerinin test edilmesi

- Her k değeri için accuracy hesaplanması

- En uygun k değerinin belirlenmesi

#### 3. Mesafe Metrikleri Karşılaştırması
- L1 (Manhattan) ve L2 (Euclidean) metriklerinin performans analizi

- Farklı k değerlerinde metrik karşılaştırması

İstatistiksel farkların değerlendirilmesi

#### 4. Scikit-learn Karşılaştırması
- Kendi implementasyonumuz ile scikit-learn KNN karşılaştırması

- Accuracy ve performans analizi

- Algoritma optimizasyon farklılıklarının incelenmesi

## Sonuçlar
Performans Metrikleri
Test Accuracy: %98.33

En iyi k değeri: 7 (%98.89 accuracy)

En iyi mesafe metriği: L2 (Euclidean)

Ortalama işlem süresi: 0.22 saniye

Karşılaştırmalı Analiz
Kendi KNN vs Scikit-learn: Aynı accuracy (%98.33)

Performans: Kendi implementasyonumuz daha hızlı (0.22s vs 3.87s)

K değeri optimizasyonu: k=7 en iyi performansı sağladı

Mesafe metrikleri: L2, L1'e göre daha iyi sonuç verdi

### Ana Bulgular
KNN algoritması MNIST veri setinde yüksek doğruluk sağlamaktadır

k parametresi seçimi model performansını kritik şekilde etkilemektedir

Euclidean mesafesi bu problem için daha uygun sonuçlar vermiştir

Kendi implementasyonumuz endüstri standardı kütüphanelerle aynı doğruluğa ulaşmıştır

## Hazırlayan
- Yiğit Buğra KÜÇÜK
- Öğrenci No: 230212048
- Bölüm: Yapay Zeka Mühendisliği
- E-posta: 230212048@ostimteknik.edu.tr