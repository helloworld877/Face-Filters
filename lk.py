import numpy as np
import cv2




# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

def klt(old_frame, new_frame):

    p0 = cv2.goodFeaturesToTrack(old_frame, mask=None,
                                **feature_params)
    
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame,
                                           new_frame,
                                           p0, None,
                                           **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d),
                        color[i].tolist(), 2)

        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)

    k = cv2.waitKey(25)
    if k == 27:
        break

    # Updating Previous frame and points
    old_frame = new_frame.copy()
    p0 = good_new.reshape(-1, 1, 2)
