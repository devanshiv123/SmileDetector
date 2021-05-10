import cv2

#Face and smile classifiers
face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector=cv2.CascadeClassifier('haarcascade_smile.xml')

#Grab video feed through webcam
webcam=cv2.VideoCapture(0)

while True:
    #Read current frame
    (successful_frame_read,frame)=webcam.read()

    #If error then abort
    if not successful_frame_read:
        break

    #Change to grayscale
    frame_grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Detect faces
    faces=face_detector.detectMultiScale(frame_grayscale)

    #Run smile detection within each face
    for (x,y,w,h) in faces:

        #Draw rectangle around faces
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,50),4)

        #get the subframe(using numpy n dimensional slicing)
        the_face=frame[y:y+h,x:x+w]

        #Change face to grayscale
        face_grayscale=cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)

        #Detect smiles
        smiles=smile_detector.detectMultiScale(face_grayscale,scaleFactor=1.7,minNeighbors=20)
        
        #for (x_,y_,w_,h_) in smiles:
            #Draw rectangle around smiles
            #cv2.rectangle(the_face,(x_,y_),(x_+w_,y_+h_),(50,50,200),4)

        #Label face as smiling
        if len(smiles)>0:
            cv2.putText(frame,'smiling',(x,y+h+40),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))

    #To show the video screen
    cv2.imshow('Smile Detector',frame)
    
    #To make video move forward by 1ms, the key waits for key press
    key=cv2.waitKey(1)
    
    if key==81 or key==113:
        break

#Cleanup code
webcam.release()
cv2.destroyAllWindows()

print('Code completed')