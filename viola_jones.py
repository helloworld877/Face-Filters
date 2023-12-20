import cv2
import numpy as np


def viola_jones(frame):
    # get classifier's path
    front_path = "./data/haarcascades/haarcascade_frontalface_default.xml"
    profile_path = "./data/haarcascades/haarcascade_profileface.xml"


    # make classifier

    front_clf = cv2.CascadeClassifier(str(front_path))
    profile_clf = cv2.CascadeClassifier(str(profile_path))


    # Initalize final rendered detection
    faces = [[]]
    
    # detect from classifier
    
    front_faces = front_clf.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    profile_faces = profile_clf.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    flipped_profile_faces = profile_clf.detectMultiScale(
        cv2.flip(frame,1) , scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Deciding which detector to use based on sum of thier corners' coordinates
    # check if not tuple as if no face detected it will output a tuple instead of [[x,y,w,h]]
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
    # for (x, y, width, height) in faces:
    #     cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
    #     frame[y:y+height, x:x+width, 0] = gray[y:y+height, x:x+width]
    #     frame[y:y+height, x:x+width, 1] = gray[y:y+height, x:x+width]
    #     frame[y:y+height, x:x+width, 2] = gray[y:y+height, x:x+width]
    
    return faces





