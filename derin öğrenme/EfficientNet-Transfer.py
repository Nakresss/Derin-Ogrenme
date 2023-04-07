#bu kodlar Jupyter içindir!
#EfficientNet'te Transfer Öğrenmenin Etkisi

from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive

#EfficientNet Kaynak Kodunun Yüklenmesi

import warnings
warnings.filterwarnings("ignore")

!pip install -U git+https://github.com/qubvel/efficientnet

import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import efficientnet.keras as enet

# CIFAR10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Sınıf vektörlerini ikili sınıf matrislerine dönüştürme
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

from keras.backend import sigmoid

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
from keras.layers impokolart Activation
get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

model = enet.EfficientNetB0(include_top=False, input_shape=(32,32,3), pooling='avg', weights='imagenet')

# 2 tam bağlantı katmanının B0'a eklenmesi
x = model.output

x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation(swish_act)(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation(swish_act)(x)

# Çıkış katmanı
predictions = Dense(10, activation="softmax")(x)

model_final = Model(inputs = model.input, outputs = predictions)

model_final.summary()

model_final.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.0001),
              metrics=['accuracy'])

mcp_save = ModelCheckpoint('/gdrive/My Drive/EnetB0_CIFAR10_TL.h5', save_best_only=True, monitor='val_acc')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)

#print("Training....")
model_final.fit(x_train, y_train,
              batch_size=32,
              epochs=10,
              validation_split=0.1,
              callbacks=[mcp_save, reduce_lr],
              shuffle=True,
              verbose=1)

acc = model_final.evaluate(x_test, y_test)

print("Test Accuracy: {}%".format(acc*100))

import seaborn as sns
from sklearn.metrics import confusion_matrix

test_pred = model_final.predict(x_test)

import numpy as np

ax = sns.heatmap(confusion_matrix(np.argmax(y_test, axis=1),np.argmax(test_pred, axis=1)), cmap="binary",annot=True,fmt="d")

def mbConv_block(input_data, block_arg):
    """Mobile Inverted Residual block along with Squeeze and Excitation block."""
    kernel_size = block_arg.kernel_size
    num_repeat= block_arg.num_repeat
    input_filters= block_arg.input_filters
    output_filters= output_filters.kernel_size
    expand_ratio= block_arg.expand_ratio
    id_skip= block_arg.id_skip
    strides= block_arg.strides
    se_ratio= block_arg.se_ratio
    # Genişleme Evresi
    expanded_filters =  input_filters * expand_ratio
    x = Conv2D(expanded_filters, 1,  padding='same',  use_bias=False)(input_data)
    x = BatchNormalization()(x)
    x = Activation(swish_activation)(x)
    # Depthwise Evrişim Evresi
    x = DepthwiseConv2D(kernel_size, strides,  padding='same',  use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(swish_activation)(x)
    # Sıkıştırma ve Çıkarma Evresi
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, expanded_filters ))(x)
    squeezed_filters = max (1, int(input_filters * se_ratio))
    se = Conv2D(squeezed_filters , 1, activation=swish_activation, padding='same')(se)
    se = Conv2D(expanded_filters, 1, activation='sigmoid', padding='same')(se)
    x = multiply([x, se])
    # Çıkış 
    x = Conv2D(output_filters, 1, padding='same', use_bias=False)
    x = BatchNormalization()(x)
    return x