import cv2
import os
import numpy as np

subjects = ["", "Gabriel", "Beatriz"]

# Detect Images Algorithm
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier('xml/lbpcascade_frontalface.xml')
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return None, None
    
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(folder_path):
    dirs = os.listdir(folder_path)
    faces = []
    labels = []

    for dirName in dirs:
        if not dirName.startswith("s"):
            continue
        label = int(dirName.replace("s", ""))
        subjectDirPath = folder_path + "/" + dirName
        subjectImagesName = os.listdir(subjectDirPath)

        for imageName in subjectImagesName:
            if imageName.startswith("."):
                continue

            imagePath = subjectDirPath + "/" + imageName
            image = cv2.imread(imagePath)
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            face, rect = detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def redim(img, larguraN):
    scale_percent = larguraN # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
