# import numpy as np
# import cv2




# # Create some random colors
# def klt(old_frame, new_frame, feature_params, lk_params):
#     color = [0, 255, 0]



#     # Create a mask image for drawing purposes
#     mask = np.zeros_like(old_frame)


#     p0 = cv2.goodFeaturesToTrack(old_frame, mask=None,
#                                 **feature_params)
    
#     # calculate optical flow
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame,
#                                            new_frame,
#                                            p0, None,
#                                            **lk_params)

#     # Select good points
#     good_new = p1[st == 1]
#     good_old = p0[st == 1]

#     # draw the tracks
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         a, b = new.ravel().astype(int)
#         c, d = old.ravel().astype(int)
#         mask = cv2.line(mask, (a, b), (c, d),
#                         color[i].tolist(), 2)

#         frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

#     img = cv2.add(frame, mask)

#     # Updating Previous frame and points
#     old_frame = new_frame.copy()
#     p0 = good_new.reshape(-1, 1, 2)


import cv2


def track_features(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, feature_points, None, **lk_params)

    # Select good points
    good_new = new_points[status == 1]
    good_old = feature_points[status == 1]

    # Draw the tracks
    
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color=(0, 255, 0), thickness=2)
        frame = cv2.circle(frame, (a, b), 5, color=(0, 255, 0), thickness=-1)

    # Update mask and feature points
    global prev_gray, feature_points
    prev_gray = gray.copy()
    feature_points = good_new.reshape(-1, 1, 2)

    return frame

# Initialize KLT parameters and video capture
# lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# cap = cv2.VideoCapture('video.mp4')

# ret, frame1 = cap.read()
# prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# good_features_to_track = cv2.goodFeaturesToTrack(prev_gray, 200, 0.3, 7)
# feature_points = np.int0(good_features_to_track)
# mask = np.zeros_like(prev_gray)

# # Outer loop for reading video
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Track features within the function
#     tracked_frame = track_features(frame)

#     cv2.imshow('Tracking', tracked_frame)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()