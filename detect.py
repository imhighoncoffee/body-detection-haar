import numpy as np
import cv2 as cv
body_cascade = cv.CascadeClassifier('haarcascade_fullbody.xml')
upper_cascade = cv.CascadeClassifier('haarcascade_upperbody.xml')
lower_cascade = cv.CascadeClassifier('haarcascade_lowerbody.xml')
img = cv.imread('kok.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
upper = upper_cascade.detectMultiScale(gray, 1.3, 5)
lower = lower_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in bodies:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]


for (x,y,w,h) in upper:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

for (x,y,w,h) in lower:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()