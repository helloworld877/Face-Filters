
import cv2
import numpy as np
front_path = "./data/haarcascades/haarcascade_frontalface_default.xml"
profile_path = "./data/haarcascades/haarcascade_profileface.xml"


# make classifier

front_clf = cv2.CascadeClassifier(str(front_path))
profile_clf = cv2.CascadeClassifier(str(profile_path))

# defining camera footage
camera = cv2.VideoCapture(0)


# render loop
faces = [[]]
while True:
    # reading form camera and detecting faces
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    front_faces = front_clf.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    profile_faces = profile_clf.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    flipped_profile_faces = profile_clf.detectMultiScale(
        cv2.flip(gray,1) , scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)


    if type(front_faces) != tuple:
        front_coords_sum = np.sum(front_faces)
    else:
        front_coords_sum = 0
    
    if type(profile_faces) != tuple:
        profile_coords_sum = np.sum(profile_faces)
    else:
        profile_coords_sum = 0
    
    if type(flipped_profile_faces) != tuple:
        flipped_coords_sum = np.sum(flipped_profile_faces)
    else:
        flipped_coords_sum = 0
        
    max_face_orientation = max(front_coords_sum, max(profile_coords_sum, flipped_coords_sum))
    
    if max_face_orientation == front_coords_sum:
        faces = front_faces
    elif max_face_orientation == profile_coords_sum:
        faces = profile_faces
    elif max_face_orientation == flipped_coords_sum:
        faces = flipped_profile_faces
        
            
    # rendering the faces
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        frame[y:y+height, x:x+width, 0] = gray[y:y+height, x:x+width]
        frame[y:y+height, x:x+width, 1] = gray[y:y+height, x:x+width]
        frame[y:y+height, x:x+width, 2] = gray[y:y+height, x:x+width]
    
    # for (x, y, width, height) in front_faces:
    #     cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
    #     frame[y:y+height, x:x+width, 0] = gray[y:y+height, x:x+width]
    #     frame[y:y+height, x:x+width, 1] = gray[y:y+height, x:x+width]
    #     frame[y:y+height, x:x+width, 2] = gray[y:y+height, x:x+width]
    
    # for (x, y, width, height) in profile_faces:
    #     cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)
    #     frame[y:y+height, x:x+width, 0] = gray[y:y+height, x:x+width]
    #     frame[y:y+height, x:x+width, 1] = gray[y:y+height, x:x+width]
    #     frame[y:y+height, x:x+width, 2] = gray[y:y+height, x:x+width]
    
    # for (x, y, width, height) in flipped_profile_faces:
    #     cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 0, 0), 2)
    #     frame[y:y+height, x:x+width, 0] = gray[y:y+height, x:x+width]
    #     frame[y:y+height, x:x+width, 1] = gray[y:y+height, x:x+width]
    #     frame[y:y+height, x:x+width, 2] = gray[y:y+height, x:x+width]

    cv2.imshow("faces", frame)

    if (cv2.waitKey(1) == ord("q")):
        break

camera.release()
cv2.destroyAllWindows()
