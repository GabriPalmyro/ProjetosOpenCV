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

# Criação do detector de faces
df = cv2.CascadeClassifier('xml/frontalface.xml')

#Carrega arquivo e converte para tons de cinza
vid = cv2.VideoCapture(0)
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    frame = redim(frame, 90) 
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = df.detectMultiScale(frame_pb, scaleFactor=1.2, minNeighbors=2)
    for (x, y, lar, alt) in faces:
        cv2.rectangle(frame, (x, y), (x + lar, y + alt), (0,255, 255), 2)

    #Exibe um frame redimensionado (com perca de qualidade)
    cv2.imshow("Encontrando faces...", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

