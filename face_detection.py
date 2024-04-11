import cv2 as cv
import numpy as np
haar_cascade = cv.CascadeClassifier(r'haar.face.xml')
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'face_trained.yml')

img=cv.imread("/Users/ali/Downloads/jasmcaus opencv-course master Resources-Faces/val/madonna/5.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
faces_rec = haar_cascade.detectMultiScale(gray, 1.1, 4)
#faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for(x,y,w,h) in faces_rec:
    faces_roi=gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)




