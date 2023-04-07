#bu kodlar Jupyter içindir!
#Veri İndirme
!wget --show-progress --continue -O /content/shakespeare.txt http://www.gutenberg.org/files/100/100-0.txt

#Veri Üretici Oluşturma

import numpy as np
import six
import tensorflow as tf
import time
import os

# Bu adres, TensorFlow'u yapılandırırken kullanacağımız TPU'yu tanımlar.
TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']

SHAKESPEARE_TXT = '/content/shakespeare.txt'

tf.logging.set_verbosity(tf.logging.INFO)

def transform(txt, pad_to=None):
  # ascii olmayan karakterleri bırakır
  output = np.asarray([ord(c) for c in txt if ord(c) < 255], dtype=np.int32)
  if pad_to is not None:
    output = output[:pad_to]
    output = np.concatenate([
        np.zeros([pad_to - len(txt)], dtype=np.int32),
        output,
    ])
  return output

def training_generator(seq_len=100, batch_size=1024):
  """A generator yields (source, target) arrays for training."""
  with tf.gfile.GFile(SHAKESPEARE_TXT, 'r') as f:
    txt = f.read()

  tf.logging.info('Input text [%d] %s', len(txt), txt[:50])
  source = transform(txt)
  while True:
    offsets = np.random.randint(0, len(source) - seq_len, batch_size)

    # Modelimiz seyrek çapraz entropi kaybı kullanmaktadır, 
    # ancak Keras, etiketlerin giriş kayıtlarıyla aynı sıraya sahip olmasını gerektirir. 
    # Bunu açıklamak için boş bir son boyut ekliyoruz.
    
    yield (
        np.stack([source[idx:idx + seq_len] for idx in offsets]),
        np.expand_dims(
            np.stack([source[idx + 1:idx + seq_len + 1] for idx in offsets]),
            -1),
    )

six.next(training_generator(seq_len=10, batch_size=1))

#Model Oluşturma
EMBEDDING_DIM = 512

def lstm_model(seq_len=100, batch_size=None, stateful=True):
  """Language model: predict the next word given the current word."""
  source = tf.keras.Input(
      name='seed', shape=(seq_len,), batch_size=batch_size, dtype=tf.int32)

  embedding = tf.keras.layers.Embedding(input_dim=256, output_dim=EMBEDDING_DIM)(source)
  lstm_1 = tf.keras.layers.LSTM(EMBEDDING_DIM, stateful=stateful, return_sequences=True)(embedding)
  lstm_2 = tf.keras.layers.LSTM(EMBEDDING_DIM, stateful=stateful, return_sequences=True)(lstm_1)
  predicted_char = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation='softmax'))(lstm_2)
  model = tf.keras.Model(inputs=[source], outputs=[predicted_char])
  model.compile(
      optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),
      loss='sparse_categorical_crossentropy',
      metrics=['sparse_categorical_accuracy'])
  return model

tf.keras.backend.clear_session()

training_model = lstm_model(seq_len=100, batch_size=128, stateful=False)

tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    training_model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))

tpu_model.fit_generator(
    training_generator(seq_len=100, batch_size=1024),
    steps_per_epoch=100,
    epochs=10,
)
tpu_model.save_weights('/tmp/bard.h5', overwrite=True)

#Model Kestirimlerini Hesaplayın
BATCH_SIZE = 5
PREDICT_LEN = 250

# Keras, durum bilgisi olan modeller için toplu boyutun önceden belirlenmesini gerektirir.
# Her seferinde bir karakterde besleyeceğimiz ve bir sonraki karakteri tahmin edeceğimiz için 
# 1'lik bir dizi uzunluğu kullanırız.

prediction_model = lstm_model(seq_len=1, batch_size=BATCH_SIZE, stateful=True)
prediction_model.load_weights('/tmp/bard.h5')

# Modeli ilk string'i kullanarak ekiyoruz. BATCH_SIZE sayısı kadar kopyalıyoruz.

seed_txt = 'Looks it not like the king?  Verily, we must go! '
seed = transform(seed_txt)
seed = np.repeat(np.expand_dims(seed, 0), BATCH_SIZE, axis=0)

# Öncelikle, modelin durumunu başlatmak için seed çalıştırın. 
prediction_model.reset_states()
for i in range(len(seed_txt) - 1):
  prediction_model.predict(seed[:, i:i + 1])

# Şİmdi kestirimleri yapmaya başlayabiliriz!
predictions = [seed[:, -1:]]
for i in range(PREDICT_LEN):
  last_word = predictions[-1]
  next_probits = prediction_model.predict(last_word)[:, 0, :]
  
  # Çıkış dağılımından örnekler
  next_idx = [
      np.random.choice(256, p=next_probits[i])
      for i in range(BATCH_SIZE)
  ]
  predictions.append(np.asarray(next_idx, dtype=np.int32))
  

for i in range(BATCH_SIZE):
  print('PREDICTION %d\n\n' % i)
  p = [predictions[j][i] for j in range(PREDICT_LEN)]
  generated = ''.join([chr(c) for c in p])
  print(generated)
  print()
  assert len(generated) == PREDICT_LEN, 'Generated text too short'

