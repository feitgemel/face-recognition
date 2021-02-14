import face_recognition
import cv2
import os
print (cv2.__version__)
import pickle

Encodings=[] # it will be array of arrays. each array will be the array of a known / trained people
Names=[]

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
        enconding=face_recognition.face_encodings(person,None,50)[0] # since it can have several faces in the image we would like to take the first
        
        # Add it to my Array
        Encodings.append(enconding) # add the known face enconding
        Names.append(name) # add the known face name 

print(Names) # print the final names trained

# We will save the data after the tarining
with open('train.pkl','wb') as f:  # f is the object     #wb is for write
    pickle.dump(Names,f)
    pickle.dump(Encodings,f)

print('End of training')
