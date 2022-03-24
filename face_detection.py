import numpy as np
import cv2

azul = (255, 0, 0)
amarelo = (0, 255, 255)
cont = 0
padding = 15

def redim(img, larguraN):
    scale_percent = larguraN # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

df = cv2.CascadeClassifier('xml/frontalface.xml')

i = cv2.imread('imagens/festa.jpg')
img = redim(i, 80)
grayscaledImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Executa a detecção
# faces = df.detectMultiScale(iPB,
#  scaleFactor = 1.2, minNeighbors = 7,
#  minSize = (30,30), flags = cv2.CASCADE_SCALE_IMAGE)
faceCoordinates = df.detectMultiScale(grayscaledImg,
		scaleFactor=1.2,
		minNeighbors=1,
		minSize=(15, 10))

croppedImgs = []

# Desenha retangulos amarelos na imagem original (colorida)
for (x, y, w, h) in faceCoordinates:
    recorte = img[y:(y + h + padding), x:(x + w + padding)]
    novoRecorte = redim(recorte, 80)
    cont = cont + 1
    cv2.rectangle(img, (x, y), (x + w, y + h), azul, 1)
    croppedImgs.append(img[y:y + h, x:x + w])

# Exibe imagem. Título da janela exibe número de faces
cv2.imshow(str(cont)+' face(s) encontrada(s)', img)
cv2.waitKey(0)

