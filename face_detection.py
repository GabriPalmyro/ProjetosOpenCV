import training
import cv2
import numpy as np

label = []
def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = cv2.imread(test_img).copy()
    print("Face Prediction Running")

    #detect face from the image
    face, rect = training.detect_face(img)
    print(len(face), "faces detected")

    label, confidence = faceRecognizer.predict(face)
    label_text = training.subjects[label]
    
    training.draw_rectangle(img, rect)
    training.draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img

print("Preparing data...")
faces, labels = training.prepare_training_data("training_data")
print("Data prepared")

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.train(faces, np.array(labels))

print("Predicting images...")

#load test images
test_img1 = "test_data/gabriel.jpg"
test_img2 = "test_data/daniela.jpg"

#perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")

#display both images
cv2.imshow(training.subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(training.subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
