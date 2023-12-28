# imports
from face_segmenter import get_corners_on_face
import numpy as np
import cv2
import dlib

# get camera footage
# capture object
cap = cv2.VideoCapture(0)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# camera capture loop
while True:
    # read the frame and show it in a window
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = get_corners_on_face(np.copy(frame))
    face_region = result[0]
    corners = result[1]
    image_out = np.copy(frame)

    faces_viola = detector(frame)
    # if (len(corners) != 0):
    # landmarks = predictor(gray_frame, face)
    # for corner in corners:
    # valid = True
    # if np.linalg.norm(corner-)
    # for x, y in corners:
    # x = np.round(x).astype(int)
    # y = np.round(y).astype(int)
    # cv2.circle(image_out, (x, y), radius=3,
    #            color=(0, 255, 0, 255), thickness=-1)

    # cv2.imshow('output', image_out)

    # press q to quit the process
    if cv2.waitKey(1) == ord('q'):

        break
# release capture device and delete all windows
cap.release()
cv2.destroyAllWindows()
