import cv2
import numpy as np
from math import hypot


def apply_cat_ears_filter(frame, landmarks):

    mustache_image = cv2.imread("face_filters/cats.png", cv2.IMREAD_UNCHANGED)
    rows, cols, _ = frame.shape

    mouth_mask = np.zeros((rows, cols), np.uint8)

    left_mouth = (landmarks.part(4).x, landmarks.part(4).y)
    right_mouth = (landmarks.part(14).x, landmarks.part(14).y)
    center_mouth = (landmarks.part(28).x, landmarks.part(28).y)

    mouth_width = int(
        hypot(left_mouth[0] - right_mouth[0], left_mouth[1] - right_mouth[1] * 1.3))

    mouth_height = int(mouth_width*1.7)

    # New Mouth position
    top_left = (int(center_mouth[0] - mouth_width / 2),
                int(center_mouth[1] - mouth_height / 2))

    mouth = cv2.resize(mustache_image, (mouth_width, mouth_height))

    mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
    _, mouth_mask = cv2.threshold(mouth_gray, 25, 255, cv2.THRESH_BINARY_INV)

    # Find the maximum value within the mask region of the original frame
    max_value = np.max(mouth_gray[mouth_mask == 255])

    # Apply a masked assignment to replace the thresholded mask with the original frame pixels
    mouth_mask = np.where(mouth_mask == 255, max_value, mouth_gray)

    # Optionally, convert back to uint8 if needed
    mouth_mask = np.uint8(mouth_mask)

    mouth_area = frame[top_left[1]: top_left[1] + mouth_height,
                       top_left[0]: top_left[0] + mouth_width]
    # mouth_area_no_mouth = cv2.bitwise_and(
    # mouth_area, mouth_area, mask=mouth_mask)

    final_mouth = np.zeros_like(mouth_area)
    for i in range(mouth_area.shape[0]):
        for j in range(mouth_area.shape[1]):
            # area included in filter
            if mouth[i][j][3] > 0:
                final_mouth[i][j] = mouth[i][j][:3]
            else:
                final_mouth[i][j] = mouth_area[i][j]

    mouth_coordinates = (top_left, mouth_height, mouth_width)

    return mouth_area, mouth, final_mouth, mouth_coordinates
