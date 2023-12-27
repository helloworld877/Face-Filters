import cv2
import numpy as np
import dlib
from math import hypot
from nose_filter import apply_nose_filter
from viola_jones import viola_jones
from glasses_filter import apply_glasses_filter
from moustache_filter import apply_mustache_filter

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

            # Nose filter
            # nose_area, nose_pig, final_nose, nose_coordinates = apply_nose_filter(frame, landmarks)

            # (top_left, nose_height, nose_width) = nose_coordinates

            # frame[top_left[1]: top_left[1] + nose_height,
            #             top_left[0]: top_left[0] + nose_width] = final_nose

            # cv2.imshow("Nose area", nose_area)
            # cv2.imshow("Nose pig", nose_pig)
            # cv2.imshow("final nose", final_nose)
            # Nose filter

            # Glasses filter
            # eyes_area, eyes, final_eyes, eyes_coordinates = apply_glasses_filter(
            #     frame, landmarks)
            # moustache filter
            eyes_area, eyes, final_eyes, eyes_coordinates = apply_mustache_filter(
                frame, landmarks)

            (top_left, eyes_height, eyes_width) = eyes_coordinates
            # temp = final_eyes.copy()

            frame[top_left[1]: top_left[1] + eyes_height,
                  top_left[0]: top_left[0] + eyes_width] = final_eyes

            # cv2.imshow("eyes area", eyes_area)
            # cv2.imshow("eyes pig", eyes)
            # cv2.imshow("final nose", final_eyes)
#            Glasses filter

            # # Mustache filter
            # mouth_area, mouth, final_mouth, mouth_coordinates = apply_mustache_filter(frame, landmarks)

            # (top_left, mouth_height, mouth_width) = mouth_coordinates

            # frame[top_left[1]: top_left[1] + mouth_height,
            #             top_left[0]: top_left[0] + mouth_width] = final_mouth

            # cv2.imshow("Mouth area", mouth_area)
            # cv2.imshow("Mouth", mouth)
            # cv2.imshow("final Mouth", final_mouth)
            # # Mustache filter

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
