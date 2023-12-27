import cv2
import numpy as np
import dlib
from math import hypot
from nose_filter import apply_nose_filter
from viola_jones import viola_jones

# Loading Camera and Nose image and Creating mask
cap = cv2.VideoCapture(0)
nose_image = cv2.imread("clown.png")
_, frame = cap.read()
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
            
            nose_area, nose_pig, final_nose, nose_coordinates = apply_nose_filter(frame, landmarks)
            
            (top_left, nose_height, nose_width) = nose_coordinates
            
            frame[top_left[1]: top_left[1] + nose_height,
                        top_left[0]: top_left[0] + nose_width] = final_nose

            cv2.imshow("Nose area", nose_area)
            cv2.imshow("Nose pig", nose_pig)
            cv2.imshow("final nose", final_nose)



        cv2.imshow("Frame", frame)



        key = cv2.waitKey(1)
        if key == 27:
            break