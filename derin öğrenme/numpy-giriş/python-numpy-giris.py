#Örnek olarak, Python'da klasik **quicksort **algoritmasının bir uygulaması:


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))

"""##Temel veri tipleri
Çoğu dil gibi, Python da tamsayılar, floats, boole ve dizeleri (strings) içeren bir dizi temel türe sahiptir. Bu veri türleri, diğer programlama dillerinde olduğu gibi biryapıya sahiptir Python'da da.
**Sayılar: **Tamsayılar (integers) ve floats diğer dillerde olduğuyla aynı, beklediğiniz gibi çalışır:
"""

x = 3
print(type(x)) # ekrana yazdır "<class 'int'>"
print(x)       # ekrana yazdır "3"
print(x + 1)   # Toplama; ekrana yazdır "4"
print(x - 1)   # Çıkarma; ekrana yazdır "2"
print(x * 2)   # Çarpma; ekrana yazdır "6"
print(x ** 2)  # Üstel; ekrana yazdır "9"
x += 1
print(x)  # ekrana yazdır "4"
x *= 2
print(x)  # ekrana yazdır "8"
y = 2.5
print(type(y)) # ekrana yazdır "<class 'float'>"
print(y, y + 1, y * 2, y ** 2) # ekrana yazdır "2.5 3.5 5.0 6.25"

"""**Booleanlar:** Python, tüm olağan operatörleri Boole mantığı için kullanır, ancak sembollerden ziyade İngilizce kelimeler kullanır (&&, ||, vb.):"""

t = True
f = False
print(type(t)) # ekrana yazdır "<class 'bool'>"
print(t and f) # Mantık VE kapısı; ekrana yazdır "False"
print(t or f)  # Mantık VEYA kapısı; ekrana yazdır "True"
print(not t)   # Mantık DEĞİL kapısı; ekrana yazdır "False"
print(t != f)  # Mantık DIŞLAYAN VEYA kapısı; ekrana yazdır "True"

"""**Dizile (Strings): **Python kolaylıkla strings işlemleri yapabilirsiniz."""

hello = 'hello'    # dizi değişkenleri tek tırnak içinde kullanılır
world = "world"    # ya da çift tırnak :)
print(hello)       # ekrana yazdır "hello"
print(len(hello))  # dizi uzunluğu; ekrana yazdır "5"
hw = hello + ' ' + world  # iki dizinn bağlanması
print(hw)  # ekrana yazdır "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # farklı formattaki dizilerin birlikte yazdırılması
print(hw12)  # ekrana yazdır "hello world 12"

s = "hello"
print(s.capitalize())  # Baş harfi büyük dizi; ekrana yazdır "Hello"
print(s.upper())       # Dizinin tüm harflerini büyük yazmak; ekrana yazdır "HELLO"
print(s.rjust(7))      # sağa yaslı yazmak ekrana yazdır "  hello"
print(s.center(7))     # Center a ortalayarak yazmak; ekrana yazdır " hello "
print(s.replace('l', '(ell)'))  # Bir alt dizinin tüm örneklerini bir diğeriyle değiştirin;
                                # ekrana yazdır "he(ell)(ell)o"

"""Python birkaç yerleşik container türü içerir: listeler, sözlükler, kümeler ve kopyalar.
**Listeler**
Bir liste, bir dizinin Python karşılığıdır, ancak yeniden boyutlandırılabilir ve farklı türde öğeler içerebilir:
"""

xs = [3, 1, 2]    # Bir liste tanımlamak
print(xs, xs[2])  # ekrana yazdır "[3, 1, 2] 2"
xs.append('bar')  # listeye bir eleman ekleme
print(xs)         # ekrana yazdır "[3, 1, 'foo', 'bar']"
x = xs.pop()      # Listenin son elemanını kaldırın ve geri alın.
print(x, xs)      # ekrana yazdır "bar [3, 1, 'foo']"

"""**Slicing:** Liste öğelerine birer birer erişmeye ek olarak, Python alt listelere erişmek için kısa bir sözdizimi sağlar; alicing olarak bilinir:"""

nums = list(range(5))     # range, tamsayıların bir listesini oluşturan yerleşik bir işlevdir.
print(nums)               # ekrana yazdır "[0, 1, 2, 3, 4]"
print(nums[2:4])          # Endeks 2'den 4'e (özel) bir dilim alır; ekrana yazdır "[2, 3]"
print(nums[2:])           # Endeks 2'den dizinin sonuna (özel) bir dilim alır; ekrana yazdır "[2, 3, 4]"
print(nums[:2])           # Endeks başlangıçtan 2'ye (özel) bir dilim alır; ekrana yazdır "[0, 1]"
print(nums[:])            # Endeks listenin tamamını alır; ekrana yazdır "[0, 1, 2, 3, 4]"
print(nums[:-1])          # Endeks 2'den 4'e (özel) bir dilim alın;; ekrana yazdır "[0, 1, 2, 3]"
nums[2:4] = [8, 9]        # Bir dilim için yeni bir alt liste atayın.
print(nums)               # ekrana yazdır "[0, 1, 8, 9, 4]"

"""**Döngüler:** Bir listenin elemanlarını şu şekilde değiştirebilirsiniz:"""

animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)

"""Döngü gövdesindeki her bir öğenin dizinine erişmek istiyorsanız, yerleşik numaralandırma işlevini kullanın: `enumerate`"""

animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))

"""**Liste anlama:** Programlama yaparken, sıklıkla bir veri türünü diğerine dönüştürmek istiyoruz. Basit bir örnek olarak, kare sayılarını hesaplayan aşağıdaki kodu dikkate alın:"""

nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)   # ekrana yazdır [0, 1, 4, 9, 16]

nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)   # ekrana yazdır [0, 1, 4, 9, 16]

nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # ekrana yazdır "[0, 4, 16]"

### **Sözlükler**
Java'daki bir haritaya  (`Map`) veya Javascript'teki bir nesneye benzer bir sözlük depoları (anahtar, değer) çiftleri. Bunu şu şekilde kullanabilirsiniz:

d = {'cat': 'cute', 'dog': 'furry'}  # Yeni bir sözlük oluştur.
print(d['cat'])       # Sözlükten bir giriş alın; ekrana yazdır "cute"
print('cat' in d)     # bir sözlüğün bir anahtarı olup olmadığını kontrol edin; ekrana yazdır "True"
d['fish'] = 'wet'     # Bir giriş ayarla
print(d['fish'])      # ekrana yazdır "wet"
# ekrana yazdır(d['monkey'])  # KeyError: 'monkey' d sözlüğünün bir anahtarı değildir
print(d.get('monkey', 'N/A'))  # Varsayılan olarak bir öğe al; ekrana yazdır "N/A"
print(d.get('fish', 'N/A'))    # Varsayılan olarak bir öğe al; ekrana yazdır "wet"
del d['fish']         # Bir öğeden bir sözlüğü kaldır
print(d.get('fish', 'N/A')) # "fish" artık bir anahtar değil; ekrana yazdır "N/A"

"""**Döngüler:** Bir sözlükteki anahtarlar üzerinden tekrarlamak kolaydır:"""

d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))

"""Anahtarlara ve bunlara karşılık gelen değerlere erişmek istiyorsanız, **`item` **yöntemini kullanın:"""

d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))

"""**Sözlük kavrayışları:** Bunlar liste kavramalarına benzer, ancak sözlükleri kolayca oluşturmanıza olanak tanır.
Örneğin:
"""

nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # ekrana yazdır "{0: 0, 2: 4, 4: 16}"

"""### Setler
Bir set, farklı unsurların sırasız bir koleksiyonudur. Basit bir örnek olarak, aşağıdakileri dikkate alın:
"""

animals = {'cat', 'dog'}
print('cat' in animals)   # Bir öğenin bir kümede olup olmadığını kontrol edin; ekrana yazdır "True"
print('fish' in animals)  # ekrana yazdır "False"
animals.add('fish')       # Bir kümeye öğe ekle
print('fish' in animals)  # ekrana yazdır "True"
print(len(animals))       # Bir kümedeki eleman sayısı; ekrana yazdır "3"
animals.add('cat')        # Zaten içinde olan bir eleman eklemek hiçbir şey değiştirmez.
print(len(animals))       # ekrana yazdır "3"
animals.remove('cat')     # Bir öğeyi bir kümeden kaldır
print(len(animals))       # ekrana yazdır "2"

"""**Döngüler:** Bir küme üzerinde yineleme, bir liste üzerinde yinelemekle aynı sözdizimine sahiptir; Ancak kümeler sırasız olduğundan, kümenin öğelerini ziyaret ettiğiniz sırayla ilgili varsayımlarda bulunamazsınız:"""

animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))

"""Listeler ve sözlükler gibi setler kullanarak kolayca set oluşturabiliriz:"""

from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # ekrana yazdır "{0, 1, 2, 3, 4, 5}"

"""### Tuples (tanımlama grupları)
Bir tuple (değişmez) sıralı bir değer listesidir. Bir tuple birçok yönden bir listeye benzer; En önemli farklılıklardan biri, sözlüklerin anahtar kelimeler olarak ve kümelerin öğeleri olarak kullanılabilmesidir; İşte basit bir örnek:
"""

d = {(x, x + 1): x for x in range(10)}  # Tuple anahtarları ile bir sözlük oluştur
t = (5, 6)        # Bir TUPLE oluştur
print(type(t))    # ekrana yazdır "<class 'tuple'>"
print(d[t])       # ekrana yazdır "5"
print(d[(1, 2)])  # ekrana yazdır "1"

"""### Fonksiyonlar
Python işlevleri, def anahtar sözcüğünü kullanarak tanımlanır. Örneğin:
"""

def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))

def hello(name, loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' % name)

hello('Bob') # ekrana yazdır "Hello, Bob"
hello('Fred', loud=True)  # ekrana yazdır "HELLO, FRED!"

"""### Sınıflar
Python'daki sınıfları tanımlamak için sözdizimi basittir:
"""

class Greeter(object):

    # Kurma işlemi
    def __init__(self, name):
        self.name = name  # Bir değişken oluştur

    # Instance method-Örnek Yöntemi
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Greeter sınıfının bir örneğini oluşturmak
g.greet()            # instance method çağır; ekrana yazdır "Hello, Fred"
g.greet(loud=True)   # instance method çağır; ekrana yazdır "HELLO, FRED!"

"""## NumPy
Numpy, Python'da bilimsel bilgi işlem için çekirdek kütüphanedir. Yüksek performanslı çok boyutlu bir dizi nesnesi ve bu dizilerle çalışmak için araçlar sağlar. MATLAB ile önceden tanışıyorsanız, bu tutorial Numpy ile çalışmaya başlamak için çok anlaşılır olacaktır.
### Diziler
Bir numpy dizisi, hepsi aynı türden bir değerler grididir ve negatif olmayan tamsayılar için bir tuple indekslenir. Boyutların sayısı dizinin sırasıdır; Bir dizinin şekli, her boyut boyunca dizinin boyutunu veren tamsayıların bir tuple'dır.
İç içe Python listelerinden `numpy` dizilerini ve köşeli parantezleri kullanarak erişim öğelerini başlatabiliriz:
"""

import numpy as np

a = np.array([1, 2, 3])   # 1 uzunluklu bir dizi vektör oluştur
print(type(a))            # ekrana yazdır "<class 'numpy.ndarray'>"
print(a.shape)            # ekrana yazdır "(3,)"
print(a[0], a[1], a[2])   # ekrana yazdır "1 2 3"
a[0] = 5                  # Dizinin bir elemanını değiştir
print(a)                  # ekrana yazdır "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # 1 uzunluklu iki dizi vektör oluştu
print(b.shape)                     # ekrana yazdır "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])   # ekrana yazdır "1 2 4"

#Numpy ayrıca dizi oluşturmak için birçok işlev sunar:

import numpy as np

a = np.zeros((2,2))   # 0'lardan oluşan bir dizi oluştur
print(a)              # ekrana yazdır "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # 1lerden oluşan bir dizi oluştur
print(b)              # ekrana yazdır "[[ 1.  1.]]"

c = np.full((2,2), 7)  # Sabit bir sayıdan oluşan dizi oluştır.
print(c)               # ekrana yazdır "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(2)         # Birim matris oluşturma
print(d)              # ekrana yazdır "[[ 1.  0.]
                      #          [ 0.  1.]]"

e = np.random.random((2,2))  # Crastgele değerlerden oluşan bir matris tanımlar
print(e)                     # Rastgele matris "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"

## Dizi  endeksleme (Array Indexing)
#Numpy dizileri dizine eklemek için çeşitli yollar sunar.
#**Slicing:** Python listelerine benzer şekilde, numpy dizileri parçalanabilir. Diziler çok boyutlu olabileceğinden, dizinin her boyutu için bir slice belirtmelisiniz:


import numpy as np

# rank=2 ve (3,4) boyutlu matris 
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# 2 satır ve 2 sütundan oluşan alt matrisi oluşturmak için slicing kullanımı; b dizinin boyutudur (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

print(a)

print(b)   # ekrana yazdır

"""Ayrıca, tamsayı indekslemeyi slice indekslemeyle karıştırabilirsiniz. Ancak, bunu yapmak orijinal diziden daha düşük bir sıra dizisi verecektir. Bunun MATLAB'ın dizi slicing işleme biçiminden oldukça farklı olduğunu unutmayın:"""

import numpy as np

# rank=2 ve (3,4) boyutlu matris 
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Dizinin özellik ve değerlerini çekmenin yolu
row_r1 = a[1, :]    # (rank 1) a dizisinin 2. satırı
row_r2 = a[1:2, :]  # (rank 2) a dizisinin 2. satırı
print(row_r1, row_r1.shape)  # ekrana yazdır "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # ekrana yazdır "[[5 6 7 8]] (1, 4)"

# Aynı ayrımı sütunlar içinde yapabiliriz.
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # ekrana yazdır "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # ekrana yazdır "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"

"""**Tamsayı dizi indeksleme:** Slicing kullanarak numpy dizileri indekslediğinizde, sonuç dizi görünümü her zaman özgün dizinin bir alt dizesi olacaktır. Tam tersine, tamsayı dizisi indeksleme, başka bir diziden verileri kullanarak rasgele diziler oluşturmanıza olanak sağlar. İşte bir örnek:"""

import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

# Tam sayı dizisi indeksleme örneği.
# Dönen dizi 3 uzunluklu olmalı
print(a[[0, 1, 2], [0, 1, 0]])  # ekrana yazdır "[1 4 5]"

# Yukarıdaki örnekle aynı şeyi verir.
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # ekrana yazdır "[1 4 5]"

"""Tamsayı dizi indekslemeyle ilgili kullanışlı bir numara, bir matrisin her satırındaki bir öğeyi seçer veya dönüştürür:"""

import numpy as np

# Elemanlarını bizim seçtiğimiz 3x3 bir matris oluşturma
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a)  # ekrana yazdır "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Bir dizi oluştur
b = np.array([0, 2, 0, 1])

# İndeksleri kullanarak her satırdaki bir elemanı seçin (b'den).
print(a[np.arange(4), b])  # ekrana yazdır "[ 1  6  7 11]"

# b'deki indeksleri kullanarak her bir satırdaki bir elemanı değiştiriniz.
a[np.arange(4), b] += 10

print(a)  # ekrana yazdır "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])

"""**Boole dizisi indeksleme:** Boole dizisi dizini, bir dizinin rasgele öğelerini seçmenizi sağlar. Sıklıkla bu tür bir endeksleme, bir koşulu karşılayan bir dizinin elemanlarını seçmek için kullanılır. İşte bir örnek:"""

import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)   # 2'den büyük olan değerleri bulun;
                     # a matrisinden 2'den büyük değerleri True olmayanları False olarak döndürür 

print(bool_idx)      # ekrana yazdır "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# bool_idx değerlerini sayısal olarak dönmesini sağlar.
print(a[bool_idx])  # ekrana yazdır "[3 4 5 6]"

# Aynı şeyi bu şekilde de yapabilirdik.
print(a[a > 2])     # ekrana yazdır "[3 4 5 6]"

"""## Veri tipleri
Her numpy dizisi aynı türden bir eleman grididır. Numpy, diziler oluşturmak için kullanabileceğiniz büyük bir sayısal veri kümesi sağlar. Numpy, bir dizi oluşturduğunuzda bir veri türünü tahmin etmeye çalışır, ancak dizileri oluşturan fonksiyonlar genellikle veri türünü açıkça belirtmek için isteğe bağlı bir argüman da içerir. İşte bir örnek:
"""

import numpy as np

x = np.array([1, 2])   # numpy da verinin tipini seçme
print(x.dtype)         # ekrana yazdır "int64"

x = np.array([1.0, 2.0])   # numpy da verinin tipini seçme
print(x.dtype)             # ekrana yazdır "float64"

x = np.array([1, 2], dtype=np.int64)   # Belirli bir veri türünü zorla.
print(x.dtype)                         # ekrana yazdır "int64"

"""## Dizi matematik
Temel matematiksel fonksiyonlar diziler üzerinde element olarak çalışır ve hem operatör aşırı yükleri hem de numpy modülündeki fonksiyonlar olarak kullanılabilir:
"""

import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Eleman toplama; 
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))

# Eleman çıkarma;
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))

# Eleman çarpım;
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))

# Eleman bölme;
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Eleman kare kök;
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))

"""MATLAB'dan farklı olarak, * matris çarpımı değil, elemansal çarpımdır. Bunun yerine, vektörlerin iç elemanlarını hesaplamak, bir vektörü matrisle çarpmak ve matrisleri çarpmak için nokta fonksiyonunu kullanırız. `dot`, hem numpy modülünde hem de dizi nesnelerinin örnek yöntemi olarak kullanılabilir:"""

import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Vektörlerin iç çarpımı
print(v.dot(w))
print(np.dot(v, w))

# Matris-Vektör çarpımı
print(x.dot(v))
print(np.dot(x, v))

# Matris-Matris çarpımı
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))

"""Numpy, diziler üzerinde hesaplamalar yapmak için birçok kullanışlı işlev sunar; En kullanışlı olanlardan biri:"""

import numpy as np

x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Tüm elemanların toplamı ekrana yazdır "10"
print(np.sum(x, axis=0))  # Tüm Sütunların Toplamı; ekrana yazdır "[4 6]"
print(np.sum(x, axis=1))  # Tüm Satırların Toplamı; ekrana yazdır "[3 7]"

"""Diziler kullanarak matematiksel fonksiyonların hesaplanması dışında, dizilerdeki verileri yeniden şekillendirmemiz veya başka şekillerde manipüle etmemiz gerekir. Bu tür işlemin en basit örneği bir matrisin aktarılmasıdır; Bir matrisi transpoze etmek için, bir dizi nesnesinin `T` özelliğini kullanın:"""

import numpy as np

x = np.array([[1,2], [3,4]])
print(x)    # ekrana yazdır "[[1 2]
            #          [3 4]]"
print(x.T)  # ekrana yazdır "[[1 3]
            #          [2 4]]"

# 1 uzunluklu dizinin transpozunun hiç birşeyi değiştirmediğini unutmayın :)
v = np.array([1,2,3])
print(v)    # ekrana yazdır "[1 2 3]"
print(v.T)  # ekrana yazdır "[1 2 3]"

"""## Broadcasting
Broadcasting, aritmetik işlemleri gerçekleştirirken numpy'nin farklı şekillerden oluşan dizilerle çalışmasını sağlayan güçlü bir mekanizmadır. Sıklıkla daha küçük bir diziye ve daha büyük bir diziye sahibiz ve daha büyük dizide bazı işlemleri gerçekleştirmek için daha küçük diziyi birden çok kez kullanmak isteriz.
Örneğin, bir matrisin her satırına sabit bir vektör eklemek istediğimizi varsayalım. Bunu böyle yapabiliriz:
"""

import numpy as np

# Vektör v matrisi x matrisinin her satırına ekleyeceğiz, sonucu matris y'de saklıyoruz.
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # X ile aynı boyuta sahip boş bir matris oluşturun.

# Vektör v, açık bir döngü ile x matrisinin her satırına ekleyin.
for i in range(4):
    y[i, :] = x[i, :] + v

# y matrisi aşağıdaki gibi olmalı
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)

"""Bu çalışıyor; Ancak matris x çok büyük olduğunda, Python'da açık bir döngüyü hesaplamak yavaş olabilir. X vektörünün matrisin her sırasına eklenmesi, dikey olarak çoklu kopyaları istifleyerek bir matris vv oluşturmaya eşdeğerdir, ardından x ve vv'nin eleman toplamıdır. Bu yaklaşımı şöyle uygulayabiliriz:"""

import numpy as np

# v vektörünü matrisin x'in her bir satırına ekleyeceğiz, sonucu y matrisinde saklayacağız.
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # v için 4 kez satır olarak kendini tekrar eden bir matris oluştur
print(vv)                 # ekrana yazdır "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
y = x + vv  # x ine vv eleman toplama işlemi yap
print(y)  # ekrana yazdır "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"

"""Numpy broadcasting, bu hesaplamayı gerçekte birden çok kopya oluşturmadan gerçekleştirmemizi sağlar."""

import numpy as np

# Vektör v matrisi x matrisinin her satırına ekleyeceğiz, sonucu matris y'de saklıyoruz.
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # broadcasting kullanarak x dizisi ile v'yi ekleyin.
print(y)  # ekrana yazdır "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"

"""İki dizinin broadcasti için şu kuralları izler:
1. Diziler aynı sıraya sahip değilse, her iki şekil aynı uzunluğa sahip oluncaya kadar alt sıra dizisinin şeklini 1 s ile hazırlayın.
2. İki dizinin, boyutta aynı boyuta sahip olması durumunda ya da dizilerden biri bu boyutta 1 boyutuna sahipse, bir boyutta uyumlu oldukları söylenir.
3. Diziler, her boyutta uyumluysa birlikte broadcast olabilir.
4. Broadcast ten sonra, her bir dizi, iki giriş dizisinin elemanlarının  maksimumlarına eşit bir şekle sahipmiş gibi davranır.
5. Bir dizinin 1 boyutunun ve diğer dizinin 1'den büyük boyutta olduğu herhangi bir boyutta, ilk dizi, bu boyut boyunca kopyalanmış gibi davranır.
Bu konuyu detaylı incelemek isterseniz: [Açıklama Dokümanı](https://http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc)
"""

import numpy as np

# Vektör dış çarpımlarını hesaplayın
v = np.array([1,2,3])  # v boyutu (3,)
w = np.array([4,5])    # w boyutu (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# Bir dış çarpım hesaplamak için, ilk önce v boyutunu (3, 1) bir sütun vektörü 
# olarak yeniden boyutlandırırız; Daha sonra v ve w'nin dış çarpımı olan bir 
# boyutlu (3, 2) çıktısını vermek için w'ye karşı broadcast yapabiliriz:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v, (3, 1)) * w)

# Bir matrisin her satırına bir vektör ekle
x = np.array([[1,2,3], [4,5,6]])
# x'in boyutu (2, 3) ve v'nin boyutu (3,) sonuçta eldeedilen boyut (2, 3),
# aşğıdaki matrisi oluşturmuş oluruz:
# [[2 4 6]
#  [5 7 9]]
print(x + v)

# Matrisin her sütununa bir vektör eklenebilir.
# x'in boyutu (2, 3) ve w'nun boyutu (2,).
# x'in transpozunu aldıktan sonra boyutu (3, 2) ve elde dilen son boyut (broadcasting) (3, 2); 
# bu sonucun transpozunu alırsak ta elde edilen sonuç boyut (2, 3) Matris x'e w vektörü eklenmiş ve 
# transpozu alınmı tekrar w eklenmiş ve tüm sonucun transpozu alınmıştır.
# [[ 5  6  7]
#  [ 9 10 11]]
print((x.T + w).T)
# Ya da bir başka çözüm w'yi yeniden boyutlandırıp bir sütun vektörü haline getirebiliriz (2, 1);
# Ardından direkt broadcast edebiliriz ve x ile toplayabiliriz. w
print(x + np.reshape(w, (2, 1)))

# Bir matrisi bir sabit sayı ile de çarpabiliriz.
# x boyutu (2, 3)
# [[ 2  4  6]
#  [ 8 10 12]]
print(x * 2)

"""## Matplotlib
Matplotlib bir çizim kütüphanesidir. Bu bölümde MATLAB'inkine benzer bir çizim sistemi sağlayan `matplotlib.pyplot` modülüne kısa bir giriş yapın.
**Çizdirme**
Matplotlib'deki en önemli işlev, 2B veriyi çizmenize izin veren çizimdir. İşte basit bir örnek:
"""

import numpy as np
import matplotlib.pyplot as plt

# x ve y koordinatlarını için bir sinüs eğrisi hesaplayın.
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

#
plt.plot(x, y)
plt.show()  # grafikleri görmek için plt.show () öğesini çağırmalısınız.

import numpy as np
import matplotlib.pyplot as plt

# x ve y koordinatlarını için bir sinüs ve kosinüs eğrisi hesaplayın.
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)


plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()

"""## Subplots"""

import numpy as np
import matplotlib.pyplot as plt

# x ve y koordinatlarını için bir sinüs ve kosinüs eğrisi hesaplayın.
y_sin = np.sin(x)
y_cos = np.cos(x)

# 2 satır 1 sütundan oluşan bir çizim ortamı oluşturur.
plt.subplot(2, 1, 1)

# birinci çizim 1. satıra yazar
plt.plot(x, y_sin)
plt.title('Sine')

# 2. satıra da 2. çizimi yazar
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Şekli ekranda göstermek için kullanılır
plt.show()