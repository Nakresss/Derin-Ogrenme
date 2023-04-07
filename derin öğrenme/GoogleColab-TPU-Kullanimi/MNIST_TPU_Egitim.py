#bu kodlar Jupyter içindir!
#MNIST Eğitimi
import numpy as np

import tensorflow as tf
import time
import os

import tensorflow.keras
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten,Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

print(tf.__version__)
print(tf.keras.__version__)

#TPU Kontrolü
try:
  device_name = os.environ['COLAB_TPU_ADDR']
  TPU_ADDRESS = 'grpc://' + device_name
  print('Found TPU at: {}'.format(TPU_ADDRESS))

except KeyError:
  print('TPU not found')

#Standart MNIST işlemleri
batch_size = 1024
num_classes = 10
epochs = 5
learning_rate = 0.001

# Giriş görüntüsü boyutları
img_rows, img_cols = 28, 28

# Veri kümesi eğitim ve test kümesi olarak iki kısıma ayrılır.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Sınıf vektörlerini ikili sınıf matrislerine dönüştürme işlemi 
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

#tf.data Kullanımı

def train_input_fn(batch_size=1024):
  # Girişleri bir veri kümesine dönüştür. 
  dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))

  # Karıştır, tekrar et ve batch (küme) örnekleri 
  dataset = dataset.shuffle(1000).repeat().batch(batch_size, drop_remainder=True)

  # veri kümesine.
  return dataset

def test_input_fn(batch_size=1024):
  # Girişleri bir veri kümesine dönüştür. .
  dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))

  # sKarıştır, tekrar et ve batch (küme) örnekleri 
  dataset = dataset.shuffle(1000).repeat().batch(batch_size, drop_remainder=True)

  # veri kümesine.
  return dataset

#Model Tanımlama
Inp = tf.keras.Input(
    name='input', shape=input_shape, batch_size=batch_size, dtype=tf.float32)

x = Conv2D(32, kernel_size=(3, 3), activation='relu',name = 'Conv_01')(Inp)
x = MaxPooling2D(pool_size=(2, 2),name = 'MaxPool_01')(x)
x = Conv2D(64, (3, 3), activation='relu',name = 'Conv_02')(x)
x = MaxPooling2D(pool_size=(2, 2),name = 'MaxPool_02')(x)
x = Conv2D(64, (3, 3), activation='relu',name = 'Conv_03')(x)
x = Flatten(name = 'Flatten_01')(x)
x = Dense(64, activation='relu',name = 'Dense_01')(x)
x = Dropout(0.5,name = 'Dropout_02')(x)

output = Dense(num_classes, activation='softmax',name = 'Dense_02')(x)


model = tf.keras.Model(inputs=[Inp], outputs=[output])
# Keras optimizer yerine tf optimizer kullanılmalıdır.

model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['acc'])

#Keras Modelinde TPU Oluşturma/Çalıştırma
tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)))

tpu_model.summary()
#Eğitim işleminde tf.data kullanımı
tpu_model.fit(
  train_input_fn,
  steps_per_epoch = 60,
  epochs=10,
)

tpu_model.save_weights('./MNIST_TPU_1024.h5', overwrite=True)

#Sonuç
tpu_model.evaluate(test_input_fn, steps = 100)