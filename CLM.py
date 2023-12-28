import cv2
import numpy as np
import dlib
from math import hypot
from viola_jones import viola_jones
from face_filters.nose_filter import apply_nose_filter
from face_filters.glasses_filter import apply_glasses_filter
from face_filters.moustache_filter import apply_mustache_filter
from face_filters.cat_ears_filter import apply_cat_ears_filter

# Loading Camera and Nose image and Creating mask
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://192.168.1.29:8080/video")
_, frame = cap.read()
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("./shape_predictor/shape_predictor_68_face_landmarks.dat")

filter_type = 3

while True:
    
    _, frame = cap.read()
    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faces_viola = viola_jones(frame)
    faces_viola = detector(frame)

    if len(faces_viola) != 0:
        for face in faces_viola:
            # (x,y,w,h) = face
            # face = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))

            landmarks = predictor(gray_frame, face)
            
            if filter_type == 0 : # beach sunglasses
                area, obj, final, coordinates = apply_glasses_filter(frame, landmarks)

            elif filter_type == 1: #clown 
                area, obj, final, coordinates = apply_nose_filter(frame, landmarks)
                
            elif filter_type == 2: #cats
                area, obj, final, coordinates = apply_cat_ears_filter(frame, landmarks)

            else: # Mustache
                area, obj, final, coordinates = apply_mustache_filter(frame, landmarks)


            (top_left, height, width) = coordinates
            frame[top_left[1]: top_left[1] + height, top_left[0]: top_left[0] + width] = final
            
            

        cv2.imshow("Frame", frame)
    else:
        cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
