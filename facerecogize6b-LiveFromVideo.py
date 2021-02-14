import face_recognition
import pafy
import cv2
import os
import pickle
import time
print(cv2.__version__)
fpsReport=0
scaleFactor=0.25

Encodings=[]
Names=[]
font=cv2.FONT_HERSHEY_SIMPLEX

with open('train.pkl','rb') as f:
    Names=pickle.load(f)
    Encodings=pickle.load(f)

print(Names)

print ('Finish load the training')
 
cam = cv2.VideoCapture('/home/eran/Desktop/pyPro/FaceRecognizer/demoImages/Video/lehaka2.mov')

#cam = cv2.VideoCapture('/home/eran/Desktop/pyPro/FaceRecognizer/demoImages/known/Eran Feit.JPG')


while True:
    _,frame=cam.read()
    # frame=cv2.resize(frame,(640,480))

    # # make the frame small by 1/4 -> after we multiple it by 4
    frameSmall=cv2.resize(frame,(0,0),fx=scaleFactor,fy=scaleFactor)
    
    # this reads in BGR and the face_recog library works in RGB
    frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    #frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    

    #print ('before facePositions action')

    # we would like to find the faces in the frame
    # the cnn is much more to power . in the Raspberri pie we need 
    #facePositions = face_recognition.face_locations(frameRGB,model='cnn') # the default model which is for lower machine is 'hog'
    facePositions = face_recognition.face_locations(frameRGB)
    print(facePositions)

    #print ('facePositions:',facePositions)
    # Now we will have all the encodings of the faces in the camera image
    allEncodings=face_recognition.face_encodings(frameRGB,facePositions)

    for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
        name='Unknown person'

        # we need to compare the face encoding to all encodings

        # face_encoding - this is a specific face inside the loop
        # Encondings - these re all the faces that was trained before 
        
        # matches will be an array of true/false of the specific face for each of the Encodings
        matches=face_recognition.compare_faces(Encodings,face_encoding) 
        print (matches)

        #print ('matches',matches)
        if True in matches:
            # getting the position index of the first in the matches true
            first_match_index=matches.index(True)
            name=Names[first_match_index]
            print(name)
        
        
        # now back to the original frame
        # put a box and the name
        top=int(top/scaleFactor)
        right=int(right/scaleFactor)
        left=int(left/scaleFactor)
        bottom=int(bottom/scaleFactor)
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
        cv2.putText(frame,name,(left,top-6),font,.75,(0,0,255),2)

    cv2.imshow('Picture',frame)
    #cv2.moveWindow('Picture',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

