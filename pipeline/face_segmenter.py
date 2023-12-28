import pathlib
import cv2
import numpy as np
from harris_corner_detector import get_good_features_to_track


def get_corners_on_face(frame):

    cascade_path = "data/haarcascades/haarcascade_frontalface_default.xml"
    # print(str(cascade_path))
    # make classifier

    clf = cv2.CascadeClassifier(str(cascade_path))

    # Reading from the camera and detecting faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(
        30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # corner array
    corners = []

    x = -1
    y = -1
    width = -1
    height = -1
    if (len(faces) != 0):
        # we found a face so we get the corners

        (x, y, width, height) = faces[0]
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        final_image = np.copy(frame)

        # Extract the face region
        face_region = gray[y:y+height, x:x+width]

        # Apply threshold within the face region
        ret, thresh = cv2.threshold(
            face_region, 125, 255, cv2.THRESH_BINARY)

        # Replace the face region in the frame with the thresholded face region
        frame[y:y+height, x:x +
              width] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_frame = np.zeros_like(frame_gray)
        face_frame[y:y+height, x:x +
                   width] = frame_gray[y:y + height, x:x+width]
        corners = get_good_features_to_track(face_frame)

    # we return the corners and coordinates
    return ((x, y, width, height), corners)
