import cv2 
import numpy as np

def apply_nose_filter(frame, landmarks):
    nose_image = cv2.imread("clown.png")
    rows, cols, _ = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)
    
    
    
    top_nose = (landmarks.part(29).x, landmarks.part(29).y)
    center_nose = (landmarks.part(30).x, landmarks.part(30).y)
    left_nose = (landmarks.part(31).x, landmarks.part(31).y)
    right_nose = (landmarks.part(35).x, landmarks.part(35).y)
    
    nose_width = int(hypot(left_nose[0] - right_nose[0],
                        left_nose[1] - right_nose[1]) * 1.7)
    nose_height = int(nose_width * 0.77)
    
    
        # New nose position
    top_left = (int(center_nose[0] - nose_width / 2),
                            int(center_nose[1] - nose_height / 2))
    bottom_right = (int(center_nose[0] + nose_width / 2),
                    int(center_nose[1] + nose_height / 2))

    