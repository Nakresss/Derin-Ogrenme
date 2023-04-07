#bu kodlar Jupyter içindir!
#Basit Bir Öğrenme Modelinde Aşırı Uydurma (Overfitting) Probleminin Çözümü: Erken Durdurma (Early Stopping)

from google.colab import drive
drive.mount('/content/drive/')

# Load libraries

from __future__ import print_function
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras. layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow  as tf
from keras.layers import *
from keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt


np.random.seed(0)

# MNIST veri kümesini indir
(x_train, y_train), (x_test, y_test) = mnist.load_data()

batch_size = 128 # Küme Boyutu
num_classes = 10 # Sınıf Sayısı
epochs = 100 # Eğitimin epoch sayısı
w_l2 = 1e-5 # Başlangıç
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
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

# sınıf vektörlerini ikili sınıf matrislerine dönüştürmek
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),  kernel_regularizer=regularizers.l2(w_l2),
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(Conv2D(64, (3, 3),  kernel_regularizer=regularizers.l2(w_l2)))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, kernel_regularizer=regularizers.l2(w_l2)))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          #callbacks=callbacks, # ERKEN DURDURMA
          verbose=1,
          validation_data=(x_test, y_test)) 



score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[1])
print('Test accuracy:', score[1])
print('val_loss:', score[1])


# Tüm veriyi history değişkeninde tut
print(history.history.keys())
# başarımları ekrana çizdir
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['test', 'val'], loc='lower right')
plt.show()
# yitimleri ekrana çizdir
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test', 'val'], loc='upper right')
plt.show()

keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50, verbose=0, mode='auto')

callbacks = [EarlyStopping(monitor='val_loss', patience=50),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

history2 = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks, # ERKEN DURDURMA
          verbose=1,
          validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[1])
print('Test accuracy:', score[1])
print('val_loss:', score[1])

# Tüm veriyi history değişkeninde tut
print(history2.history.keys())
# başarımları ekrnaa çiz
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['test', 'val'], loc='lower right')
plt.show()
# yitimleri ekrana çizdir
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test', 'val'], loc='upper right')
plt.show()