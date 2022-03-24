from math import fabs
from numpy.linalg import norm
import numpy as np
import cv2 as cv

azul = (255, 0, 0)
verde = (0, 255, 0)
vermelho = (0, 0, 255)
amarelo = 	(255, 117, 24)
fonte = cv.FONT_HERSHEY_SIMPLEX
pad = 60

dados = 0
pontuacaoDados = 0

img = cv.imread('imagens/dados.jpg')

grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurImage = cv.blur(grayImage, (3, 3))
ret, thresh1 = cv.threshold(blurImage, 170, 255, cv.THRESH_BINARY)
bordas = cv.Canny(thresh1, 80, 230)
  
contours, hierarchy = cv.findContours(bordas, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

diceRects = []

for countor in contours:
  #! For each contour, search the minimum area rectangle
  rect = cv.minAreaRect(countor)
  
  #! Devemos apenas processar os retang que são quadrados e do tamanho correto
  if rect[1][1] != 0:
    aspect = fabs((rect[1][0] / rect[1][1]) - 1)
  else:
    aspect = 0

  rectArea = rect[1][0] * rect[1][1]
  box = cv.boxPoints(rect)
  box = np.int0(box)

  #! Verificar se é um dado mesmo
  if (aspect < 0.3) and (rectArea > 12000 and rectArea < 35000):
    #! Verificar se o dado está repetido
    process = True
    for diceRect in diceRects:
      dist = norm(np.asarray(rect[0]) - np.asarray(diceRect[0]))
      if dist < 10:
        process = False
        break
    
    if process:
      diceRects.append(rect)
      cv.drawContours(img,[box],0,verde,3)
      dados += 1

diceCounts = [0, 0, 0, 0, 0, 0]
print("Quant de dados", dados)

for dice in diceRects:
  #! Extrair imagem do dado
  rectCenter = np.asarray(dice[0])
  rectSize = np.asarray(dice[1])
  rectRot = dice[2]
  x = int(rectCenter[0])
  y = int(rectCenter[1])
  h = int(rectSize[0])
  w = int(rectSize[1])
  # print(rectCenter, rectSize, rectRot)
  rotation = cv.getRotationMatrix2D(rectCenter, rectRot, 1.0)
  imgRotated = cv.warpAffine(grayImage, rotation, (grayImage.shape[1], grayImage.shape[0]), cv.INTER_CUBIC)
  imgCropped = imgRotated[(y-h+70):(y+h-70), (x-w+70):(x+w-80)]
  # imgCropped = cv.getRectSubPix(imgRotated, np.size(rectSize[0], rectSize[1]), rectCenter)

  #! Processar face de cima do dado
  croppedRet, croppedTresh = cv.threshold(imgCropped, 170, 255, cv.THRESH_BINARY)
  bordas = cv.Canny(croppedTresh, 80, 180)
  dieContours, dieHierarchy = cv.findContours(bordas, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

  dotsRects = []
  for dieContour in dieContours:
    dotRect = cv.minAreaRect(dieContour)

    if dotRect[1][1] != 0:
      aspect = fabs((dotRect[1][0] / dotRect[1][1]) - 1)
    else:
      aspect = 1

    dotRectArea = dotRect[1][0] * dotRect[1][1]
    box = cv.boxPoints(dotRect)
    box = np.int0(box)

    #! Verificar se é um circulo do dado
    if (aspect < 0.45) and (dotRectArea > 50 and dotRectArea < 2000):
    #! Verificar se o circulo do dado está repetido
      process = True
      for rect in dotsRects:
        dist = norm(np.asarray(dotRect[0]) - np.asarray(rect[0]))
        if dist < 10:
          process = False
          break 
      if process:
        dotsRects.append(dotRect)
        pontuacaoDados += 1

cv.putText(img,"Quantidade de dados: "+str(dados), (20, 30), cv.FONT_HERSHEY_DUPLEX, 0.8, azul, 1, cv.LINE_AA)
cv.putText(img,"Pontuacao dos dados: "+str(pontuacaoDados), (20, 60), cv.FONT_HERSHEY_DUPLEX, 0.6, vermelho, 1, cv.LINE_AA)
cv.imshow("Image", img)

cv.waitKey(0)
cv.destroyAllWindows()