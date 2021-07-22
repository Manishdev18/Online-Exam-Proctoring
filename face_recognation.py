import cv2 as cv
import face_recognition
import os
import numpy as np

from face_recognition.api import face_encodings
path = "imgfile"
images = []
className = []

mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curimg = cv.imread(f'{path}/{cl}')
    images.append(curimg)
    className.append(os.path.splitext(cl)[0])


def findencoding(images):
    encodelist =[]
    for img in images:
        cimg = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknown = findencoding(images)
print(len(encodelistknown))

cap = cv.VideoCapture(0)

while True:
    sucess , frame = cap.read()
    #frame = cv.resize(frame,(0,0),None,0.25,0.25)
    frame = cv.cvtColor(frame ,cv.COLOR_BGR2RGB)
    cfaceloc = face_recognition.face_locations(frame)
    cfaceloc = face_recognition.face_locations(frame)
    vencode = face_recognition.face_encodings(frame,cfaceloc)

    for encodeface,faceloc in zip(vencode,cfaceloc):
        matches = face_recognition.compare_faces(encodelistknown,encodeface)
        facedis = face_recognition.face_distance(encodelistknown,encodeface)
        
        matchindex = np.argmin(facedis)

        if matches[matchindex]:
            name = className[matchindex].upper()
            y1,x2,y2,x1=faceloc
            #y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv.rectangle(frame,(x1,y2+35),(x2,y2),(0,255,0),cv.FILLED)
            cv.putText(frame,name,(x1-4,y2+24),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

    cv.imshow('webcam',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    

