# imports
from face_segmenter import get_corners_on_face
import numpy as np
import cv2

# get camera footage
# capture object
cap = cv2.VideoCapture("http://192.168.1.29:8080/video")

# camera capture loop
while True:
    # read the frame and show it in a window
    ret, frame = cap.read()

    result = get_corners_on_face(np.copy(frame))
    face_region = result[0]
    corners = result[1]
    image_out = np.copy(frame)
    if (len(corners) != 0):
        for x, y in corners:
            x = np.round(x).astype(int)
            y = np.round(y).astype(int)
            cv2.circle(image_out, (x, y), radius=3,
                       color=(0, 255, 0, 255), thickness=-1)
    cv2.imshow('output', image_out)

    # press q to quit the process
    if cv2.waitKey(1) == ord('q'):

        break
# release capture device and delete all windows
cap.release()
cv2.destroyAllWindows()
