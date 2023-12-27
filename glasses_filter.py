import cv2 
import numpy as np
from math import hypot

def apply_glasses_filter(frame, ladmarks):
    glasses_image = cv2.imread("clown.png")
        rows, cols, _ = frame.shape