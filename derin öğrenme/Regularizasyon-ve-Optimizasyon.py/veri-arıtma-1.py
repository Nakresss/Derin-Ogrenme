from google.colab import drive
drive.mount('/content/drive/')

"""**Drive da dosya konumlandırmayı yapma işlemleri**"""

!ls

!ls drive

import os
os.chdir("/content/drive/My Drive/Udemy_DerinOgrenmeyeGiris/Regularizasyon ve Optimizasyon")

!ls

"""**Paketlerin Yüklenmesi**"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

"""### Kullanılacak veri artırma yönteminin seçilmesi ve hiper parametrelerinin belirlenmesi işlemleri"""

# kullanılacak veri artırma tekniklerini tanımla
datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest')

"""**Artırma işlemi yapılacak olan görüntünün dosyadan okunması işlemi**"""

# tek bir resmi yükle
# farklı resim dosyasını da deneyebilirsiniz

img = load_img('araguler.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

"""**Giriş görüntüsünden kaç tane üretilmesini istediğimizle ilgili oluşturduğumuz `for` döngüsü ve sonuçların ilgili formatla kaydedilip, dosyaya yazdılırması işlemleri.**"""

# tek resimden 50 tane farklı resim üret ve Artirilmis_Veri klasörüne .jpeg formatında kaydet.
i = 0

for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/content/drive/My Drive/Udemy_DerinOgrenmeyeGiris/Regularizasyon ve Optimizasyon/Artirilmis_Veri', 
                          save_format='jpg'):
    i += 1
    if i > 50:
        break