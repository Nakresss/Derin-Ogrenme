from google.colab import drive
drive.mount('/content/drive/')

"""**Drive da dosya konumlandırmayı yapma işlemleri**"""

!ls

!ls drive

import os
os.chdir("/content/drive/My Drive/Udemy_DerinOgrenmeyeGiris/Regularizasyon ve Optimizasyon")

!ls

"""### CIFAR 10 veri kümesi ile veri artırma örnekleri ve görselleştirme"""

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
import numpy as np

# Veriyi indirme ve eğitim test kümelerini rastgele oluşturma

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255

"""# Orijinal Görüntüler"""

datagen = ImageDataGenerator()
datagen.fit(X_train)

for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=4, seed=499):
  for i in range(0,4):
    pyplot.subplot(220 +1 +i)
    pyplot.imshow(X_batch[i])
  pyplot.show()
  break

"""# DÖNDÜRME"""

datagen = ImageDataGenerator(rotation_range=359)
datagen.fit(X_train)

for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=4, seed=499):
  for i in range(0,4):
    pyplot.subplot(220 +1 +i)
    pyplot.imshow(X_batch[i])
  pyplot.show()
  break

"""# DİKEYDE KAYDIRMA"""

datagen = ImageDataGenerator(height_shift_range=0.5)
datagen.fit(X_train)

for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=4, seed=499):
  for i in range(0,4):
    pyplot.subplot(220 +1 +i)
    pyplot.imshow(X_batch[i])
  pyplot.show()
  break

"""# YATAYDA SİMETRİĞİNİ ALMA"""

datagen = ImageDataGenerator(horizontal_flip=True)
datagen.fit(X_train)

for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=4, seed=499):
  for i in range(0,4):
    pyplot.subplot(220 +1 +i)
    pyplot.imshow(X_batch[i])
  pyplot.show()
  break