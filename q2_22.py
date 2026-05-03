from utils import *
import numpy as np
from PIL import Image
from Kmeans import KMeans
from Kmeans import get_colors, distance

Path_to_img = 'C:\\Users\\Hugo Aguilar\\Desktop\\Practica2\\images\\4solid_colors.jpg'
img = Image.open(Path_to_img)
img = img.convert('RGB')
print("Imatge carregada correctament")
print("Mida:", img.size)

km = KMeans(img, K=4, options={'km_init': 'first'})
print("KMeans creat")
print("Shape de X:", km.X.shape)

km.fit()
print("Fit executat")
print("Centroids:", km.centroids)
print(get_colors(km.centroids))

img_mono = Image.open('C:\\Users\\Hugo Aguilar\\Desktop\\Practica2\\images\\gris.png')
img_mono = img_mono.convert('RGB')

# Pregunta 2 - imatge monocroma K=1 random
km2 = KMeans(img_mono, K=1, options={'km_init': 'random'})
km2.fit()
print("P2 centroids:", km2.centroids)

img_shirt = Image.open('C:\\Users\\Hugo Aguilar\\Desktop\\Practica2\\images\\samarreta.jpg')
img_shirt = img_shirt.convert('RGB')

# Pregunta 3 - samarreta K=2 WCD
km3 = KMeans(img_shirt, K=2, options={'km_init': 'first'})
km3.fit()
print("P3 WCD:", km3.withinClassDistance())

img_mono = Image.open('C:\\Users\\Hugo Aguilar\\Desktop\\Practica2\\images\\gris.png')
img_mono = img_mono.convert('RGB')

# Pregunta 4 - monocroma K=1 WCD
km4 = KMeans(img_mono, K=1, options={'km_init': 'first'})
km4.fit()
print("P4 WCD:", km4.withinClassDistance())

img_4colors = Image.open('C:\\Users\\Hugo Aguilar\\Desktop\\Practica2\\images\\4solid_colors.jpg')
img_4colors = img_4colors.convert('RGB')

Path_to_img = 'C:\\Users\\Hugo Aguilar\\Desktop\\Practica2\\images\\4solid_colors.jpg'
img = Image.open(Path_to_img)
img = img.convert('RGB')

km = KMeans(img, K=4, options={'km_init': 'first'})
km.fit()
print("Centroids:", km.centroids)
print("Colors:", get_colors(km.centroids))

# Pregunta 6 - find_bestK
km6 = KMeans(img_4colors, K=1, options={'km_init': 'first'})
km6.find_bestK(10)
print("P6 best K:", km6.K)