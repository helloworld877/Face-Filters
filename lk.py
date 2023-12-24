import numpy as np
import cv2
from PIL import Image


# Take first frame and find corners in it

# Create a mask image for drawing purposes


def klt(frame,action_frame,old_gray,p0,lk_params):
    # calculate optical flow
    if action_frame.shape[0] > old_gray.shape[0] or action_frame.shape[1] > old_gray.shape[1]:
        action_frame = cv2.resize(action_frame,old_gray.shape) 
    if action_frame.shape[0] < old_gray.shape[0] or action_frame.shape[1] < old_gray.shape[1]:
        old_gray = cv2.resize(old_gray,action_frame.shape) 
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                           action_frame,
                                           p0, None,
                                           **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        cv2.line(frame, (a, b), (c, d),
                        (0, 255, 0), 2)

        cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    # Updating Previous frame and points
    old_gray = action_frame.copy()
    p0 = good_new.reshape(-1, 1, 2)
    
    return frame
