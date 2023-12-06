# imports
import numpy as np
import cv2 
from skimage import data
from skimage.color import rgb2gray
from skimage import filters

def process_frame(frame):
    # Convert the frame to grayscale
    gray_frame = rgb2gray(frame)
    
    # Apply a filter (e.g., edge detection)
    edges = filters.sobel(gray_frame)
    
    # Return the processed frame
    return edges

# capture object
cap=cv2.VideoCapture(0)

# camera capture loop
while True:
    # read the frame and show it in a window
    ret,frame=cap.read()

    # cv2.imshow('frame',frame)
    
    # get greyscale image
    # cv2.imshow('frame',rgb2gray(frame))
    cv2.imshow('video gray', process_frame(frame))
    cv2.imshow('video original', frame)


    # press q to quit the process
    if cv2.waitKey(1)==ord('q'):

        break
# release capture device and delete all windows
cap.release()
cv2.destroyAllWindows()