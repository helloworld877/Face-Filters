import cv2
import numpy as np
from viola_jones import viola_jones

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
    
    camera = cv2.VideoCapture(0)
    while True:
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = viola_jones(gray)
        
        viola_jones_face = np.zeros(gray.shape)
        
        (x, y, width, height) = faces[0]
        # cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        viola_jones_face[y:y+height, x:x+width] = gray[y:y+height, x:x+width]
        
        feature_points = cv2.goodFeaturesToTrack(viola_jones_face, 200, 0.3, 7)
        feature_points = np.int0(feature_points) + np.array([x, y])
        
        
        cv2.imshow("faces", frame)

        if (cv2.waitKey(1) == ord("q")):
            break
        
    camera.release()
    cv2.destroyAllWindows()


main()