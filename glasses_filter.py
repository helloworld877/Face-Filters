import cv2
import numpy as np
from math import hypot


def apply_glasses_filter(frame, landmarks):
    glasses_image = cv2.imread(
        "beach_sunglasses.png", cv2.IMREAD_UNCHANGED)
    rows, cols, _ = frame.shape

    eyes_mask = np.zeros((rows, cols), np.uint8)

    # top_left_glasses = (landmarks.part(18).x, landmarks.part(18).y)
    # top_right_glasses = (landmarks.part(27).x, landmarks.part(27).y)
    left_glasses = (landmarks.part(37).x, landmarks.part(37).y)
    right_glasses = (landmarks.part(46).x, landmarks.part(46).y)
    center_glasses = (landmarks.part(28).x, landmarks.part(28).y)

    glasses_width = int(hypot(
        left_glasses[0] - right_glasses[0], left_glasses[1] - right_glasses[1] * 1.7))

    glasses_height = int(glasses_width * 0.77)

    # New glasses position
    top_left = (int(center_glasses[0] - glasses_width / 2),
                int(center_glasses[1] - glasses_height / 2))

    eyes = cv2.resize(glasses_image, (glasses_width, glasses_height))
    # print(eyes.shape)

    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
    _, eyes_mask = cv2.threshold(eyes_gray, 25, 255, cv2.THRESH_BINARY_INV)

    eyes_area = frame[top_left[1]: top_left[1] + glasses_height,
                      top_left[0]: top_left[0] + glasses_width]

    # print(eyes_area.shape)
    final_eyes = np.zeros_like(eyes_area)
    for i in range(eyes_area.shape[0]):
        for j in range(eyes_area.shape[1]):
            # area included in filter
            if eyes[i][j][3] > 0:
                final_eyes[i][j] = eyes[i][j][:3]
            else:
                final_eyes[i][j] = eyes_area[i][j]

    # eyes_area_no_eyes = cv2.bitwise_and(eyes_area, eyes_area, mask=eyes_mask)

    # final_eyes = np.zeros_like(eyes_area_no_eyes)

    # final_eyes = cv2.add(eyes_area_no_eyes, eyes[:, :, :3])

    eyes_coordinates = (top_left, glasses_height, glasses_width)

    return eyes_area, eyes, final_eyes, eyes_coordinates
