from google.colab import drive
drive.mount('/gdrive/')

from keras.applications.resnet import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from io import BytesIO
import os
import requests

model = ResNet50(weights="imagenet")
layers = dict([(layer.name, layer.output) for layer in model.layers])
model.summary()

# MODELDEKİ TOPLAM PARAMETRE SAYISINI EKRANA YAZDIR
model.count_params() 

def prepare_image(image, target):
	# giriş görüntüsünü yeniden boyutlandırma ve ön işlemerin yapılması
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# işlenmiş görüntüyü alma
	return image

#@title Görüntünün URL'sini Yapıştırın { vertical-output: true }
ImageURL = "https://www.kopekegitimleri.com/wp-content/uploads/2019/07/cavalier-king-charles-spaniel-770x433.jpg\"" #@param {type:"string"}

#ImageURL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQlfqFtPDFn8uLt5A2kR0nBl9NQ9IfaRJmxTA&usqp=CAU"
#https://3.bp.blogspot.com/-u2EcSH2R3aM/VM69jPZvvOI/AAAAAAAAYzk/xmjSdaDD06o/s1600/mercan_resif.jpg
response = requests.get(ImageURL)
image = Image.open(BytesIO(response.content))

# root = 'drive/My Drive/'
# image_path = root+ 'Olips.png'
# image = Image.open(image_path)
# image = image.resize((224, 224))
# image
# Görüntüyü diziye çevir
# x = np.asarray(image, dtype='float32')
# Dizi listesine çevir
# x = np.expand_dims(x, axis=0)
# Giriş görüntüsünü eğitim setine uygun şekilde ön işlemleri yap 
# x = preprocess_input(x)
#preds = model.predict(x)
#print('Predicted:', decode_predictions(preds, top=3)[0])
#print(decode_predictions(preds, top=1)[0][0][1])

data = {"success": False}

pre_image = prepare_image(image, target=(224, 224)) # 224 x 224 boyutlu hale getir

preds = model.predict(pre_image) # Kesirim modeline ön işlemden geçmiş görüntüyü uygula

results = imagenet_utils.decode_predictions(preds) #kestirim
data["predictions"] = []


for (imagenetID, label, prob) in results[0]: # ImageNet veri kümseinden etiket, olasılık ve kestrim sonucunu al
  r = {"label": label, "probability": float(prob)}
  data["predictions"].append(r)
  
data["success"] = True

print(data)

print("Sınıflandırma tahmini en yüksek olan {0} oranıyla {1}'dır.".format(data["predictions"][0]["probability"],data["predictions"][0]["label"])) 
# En yüksek olasılıklı sonucu ekrana yazdır