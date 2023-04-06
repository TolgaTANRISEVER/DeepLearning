# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:40:36 2023

@author: Tolga
"""


#hebb
import numpy as np

# Örnek veri seti (AND kapısı)
X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
Y = np.array([1, -1, -1, -1])

# Ağırlık matrisi başlangıç değerleri
w = np.array([0.1, 0.2])

# Hebb öğrenme algoritması
for i in range(len(X)):
    y_pred = np.sign(np.dot(w, X[i]))  # Tahmin
    w += 0.1 * y_pred * X[i] * Y[i]  # Ağırlık güncelleme

# Sonuçları yazdırma
print("Ağırlık matrisi: ", w)
print("Tahminler: ", np.sign(np.dot(X, w)))
#%%adaline 
import numpy as np

class Adaline:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta  # Öğrenme oranı
        self.n_iter = n_iter  # Tekrar sayısı
        self.random_state = random_state  # Rastgelelik ayarı

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    
    
    
    
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Iris veri setini yükle
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Özellikleri standartlaştır
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

# Adaline modelini eğit
ada = Adaline(eta=0.01, n_iter=50, random_state=1)
ada.fit(X_std, y)

y_pred = ada.predict(X_std)
print('Yapay sinir ağı modelinden doğruluk oranı: %.2f' % accuracy_score(y, y_pred))
#%%
import numpy as np

class Perceptron(object):
    """
    Perceptron sınıfı
    
    Parametreler:
    --------------
    eta : float
        Öğrenme oranı (0.0 ile 1.0 arasında)
    n_iter : int
        Veri kümesi üzerinde geçirilecek döngü sayısı
    random_state : int
        Ağırlıkların başlangıç değerlerinin rastgele atanması için bir başlangıç durumu belirler.
    
    Atributeler:
    -----------
    w_ : 1d-array
        Öğrenme için ağırlıklar
    errors_ : list
        Her epoch'ta gözlenen sınıflandırma hataları
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Veri kümesini uygun hale getirip ağırlıkları öğrenmek için fit fonksiyonu
        
        Parametreler:
        -------------
        X : {array-like}, shape = [n_samples, n_features]
            Veri kümesinin özelliklerini içeren matris
        y : {array-like}, shape = [n_samples]
            Hedef sınıflar
            
        Return:
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Net girişi hesaplar"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Sınıf etiketini döndürür"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Iris veri setini yükle
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Eğitim ve test kümelerini ayır
#Eğitim ve test kümelerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

#Özellik ölçeklendirme (Standartlaştırma)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#PERCEPTRON nesnesi oluştur
ppn = Perceptron(eta=0.1, n_iter=100)

#Veri kümesine uyarla
ppn.fit(X_train_std, y_train)

#Test verisi üzerinde tahmin yap
y_pred = ppn.predict(X_test_std)
print('Hatalı sınıflandırmalar: %d' % (y_test != y_pred).sum())