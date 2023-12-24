import cv2
import numpy as np
from viola_jones import viola_jones
from lk import klt



def main():
    # KLT parameters
    # params for corner detection
    feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))
    prev_gray = None
    feature_points = None
    mask = None
    
    camera = cv2.VideoCapture(0)
    _,first_frame = camera.read()
    viola_first_face = viola_jones(first_frame)
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    while len(viola_first_face) <= 0:
        viola_first_face = viola_jones(first_frame)
    (x_first,y_first,w_first,h_first) = viola_first_face[0]
    viola_gray_first = first_gray[y_first:y_first+h_first, x_first:x_first+w_first]
    while True:
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = viola_jones(gray)
        if len(faces) > 0:
            (x, y, width, height) = faces[0]
            # cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            viola_jones_face = gray[y:y+height, x:x+width]
            
            # feature_points = cv2.goodFeaturesToTrack(viola_jones_face, 200, 0.3, 7)
            # feature_points = np.int0(feature_points) + np.array([x, y])

            # Initialize feature points in the first frame or after face loss
            feature_points = cv2.goodFeaturesToTrack(viola_jones_face, **feature_params)
            feature_points = np.float32(feature_points) + np.float32(np.array([x, y]))


            tracked_frame = klt(frame,viola_jones_face,viola_gray_first,feature_points,lk_params)

        else:
            tracked_frame = frame  # No faces detected, display original frame
        
        cv2.imshow('Tracking', tracked_frame)
        # cv2.imshow("faces", frame)
        if (cv2.waitKey(1) == ord("q")):
            break
    camera.release()
    cv2.destroyAllWindows()


main()