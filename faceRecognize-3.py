import face_recognition
import cv2
import os
print (cv2.__version__)

Encodings=[] # it will be array of arrays. each array will be the array of a known / trained people
Names=[] # this is an array for the names of the known people

# training the known images
image_dir = '/home/eran/Desktop/pyPro/FaceRecognizer/demoImages/known'
for root,dirs, files in os.walk(image_dir):
    print ('files:',files)
    for file in files:
        path=os.path.join(root,file)
        print(path)
        name=os.path.splitext(file)[0] # it will get the name without the .jpg
        print(name)
        # load the image with the face_recognision library
        person=face_recognition.load_image_file(path)
        enconding=face_recognition.face_encodings(person)[0] # since it can have several faces in the image we would like to take the first
        
        # Add it to my Array
        Encodings.append(enconding) # add the known face enconding
        Names.append(name) # add the known face name 

print('====================================')
print('Names: ' , Names)
print('====================================')

font=cv2.FONT_HERSHEY_SIMPLEX

# loading the test image
testImage=face_recognition.load_image_file('/home/eran/Desktop/pyPro/FaceRecognizer/demoImages/unknown/test3.jpg')

# locate the faces coordinates in the test image and put in array
facePositions=face_recognition.face_locations(testImage)

# make an array of enconding of all the faces in the test image
AllEncodings=face_recognition.face_encodings(testImage,facePositions)

# since we need to display in open CV , we need to convert it to BGR color
testImage=cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)

# for each face we would like to see if it match for the array of the known training encondings
for (top,right,bottom,left),face_encoding_index in zip(facePositions,AllEncodings):
    # the default is it does not recognize anyone
    name='Unknown Person'
    matches=face_recognition.compare_faces(Encodings,face_encoding_index) # matches is an array of true and false
    if True in matches:
        first_match_index=matches.index(True) # what is the position it was found
        name=Names[first_match_index] # grab the name of the True index
    cv2.rectangle(testImage,(left,top),(right,bottom),(0,0,255),2)
    cv2.putText(testImage,name,(left,top-6),font,0.75,(0,255,255),2)

cv2.imshow('Picture',testImage)
cv2.moveWindow('Picture',0,0)

if cv2.waitKey(0)==ord('q'):
    cv2.destroyAllWindows()


