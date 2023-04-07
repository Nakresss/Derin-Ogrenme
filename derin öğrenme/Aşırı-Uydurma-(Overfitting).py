#bu kodlar Jupyter içindir!
# Google Colab Kimlik Doğrulama İşlemleri
from google.colab import drive
drive.mount('/content/drive/')

from __future__ import print_function
import numpy as np
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

# İki sınıflı rastgele dağılımlı veri kümesi üretme (moons)
from sklearn.datasets import make_moons
from matplotlib import pyplot
from pandas import DataFrame
# veri setini 2 boyutlu düzlemde oluşurma, örnek sayısı, gürültü
X, y = make_moons(n_samples=100, noise=0.2, random_state=1)
# verilerin görselleştirilmesi
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'orange', 1:'magenta'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()


# Veri kümesini eğitim ve test olarak iki parçaya ayırma
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

# Basit bir çok katmanlı sinir ağı (MLP) oluşturma
model = Sequential()
# 2 girişli 500 nöronlu gizli katman ve aktivasyon fonksiyonu ReLU
model.add(Dense(500, input_dim=2, activation='relu'))
# Çıkış katmanında tek nöron ve sigmoid fonksiyonu kullanılmaktadır
model.add(Dense(1, activation='sigmoid'))
# İkili çaprazentropi ile yitim değeri hesaplanıyor ve adam optimizasyonu ile hata minimize ediliyor. Başarım metriği olarak doğruluk kullanılıyor.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Oluşturulan sinir ağı modeli: 2 girişli 500 nöronlu gizli katman ve aktivasyon fonksiyonu ReLU

# Modeli 4000 epoch için eğit
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0)

# Modelin eğitim ve test başarımlarını hesapla ve ekrana yazdır.
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# Eğitim değerli history değişkeninde tutulmuştu tüm epochlar için bunu ekrana çizdirme işlemi
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# MLP modelini tekrar oluştur ve bu kez içine erken durdurma adımını da ekle
from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
# 2 boyutta veri setini oluşturma
X, y = make_moons(n_samples=100, noise=0.2, random_state=1)
# veri setini parçalara ayırma
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# Modeli tanımlama
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# BASİT BİR ERKEN DURDURMA İŞLEMİ
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1)
# ERKEN DURDURMA SONUCU EN İYİ BAŞARIMIN MODEL BAŞARIMI OLARAK KAYDEDİLMESİ
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
# Modeli erken durdurma ve en iyi başarımı kaydetme callback parametrelerini ekleyerek tekrar eğitme
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0, callbacks=[es, mc])
# Test ve Eğitim başarılarının hesaplanması
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# Eğitim ve test sonuçlarının ekrana yazılması ve çizdirilmesi
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

