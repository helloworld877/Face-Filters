# imports
from face_segmenter import get_corners_on_face
import numpy as np
import cv2
import dlib
import math
from math import sqrt

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
    for face in faces_viola:
        if (len(corners) != 0):
            landmarks = predictor(gray_frame, face)
            filtered_corners = []
            for l in landmarks.parts():
                
                closest_corner = min(corners, key=lambda c: sqrt((c[0]- l.x)**2 + (c[1]-l.y)**2))
                filtered_corners.append(closest_corner)
                # corners.remove(closest_corner)
                if len(filtered_corners) == 68:
                    break
                
            updated_clm_points = []
            for i, clm_point in enumerate(landmarks.parts()):
                updated_x = 0.5 * clm_point.x + 0.5 * filtered_corners[i][0]
                updated_y = 0.5 * clm_point.y + 0.5 * filtered_corners[i][1]
                updated_clm_points.append((updated_x, updated_y))

            for n in range(len(updated_clm_points)):
                x = updated_clm_points[n][0]
                y =  updated_clm_points[n][1]
                cv2.circle(image_out, (int(x), int(y)), 1, (0, 0, 255), -1)
            
            

    # for corner in corners:
    # valid = True
    # if np.linalg.norm(corner-)
    # for x, y in corners:
    # x = np.round(x).astype(int)
    # y = np.round(y).astype(int)
    # cv2.circle(image_out, (x, y), radius=3,
    #            color=(0, 255, 0, 255), thickness=-1)

    cv2.imshow('output', image_out)

    # press q to quit the process
    if cv2.waitKey(1) == ord('q'):

        break
# release capture device and delete all windows
cap.release()
cv2.destroyAllWindows()
