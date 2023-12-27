import pathlib
import cv2
import numpy as np

cascade_path = pathlib.Path(cv2.__file__).parent.absolute(
) / "data/haarcascades/haarcascade_frontalface_default.xml"

print(cascade_path)

# make classifier

clf = cv2.CascadeClassifier(str(cascade_path))

# defining camera footage
camera = cv2.VideoCapture("http://192.168.1.102:8080/video")


# render loop

while True:
    # Reading from the camera and detecting faces
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(
        30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Rendering the faces and applying threshold within the face region
    if (len(faces) != 0):

        (x, y, width, height) = faces[0]
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)

        # Extract the face region
        face_region = gray[y:y+height, x:x+width]

        # Apply threshold within the face region
        ret, thresh = cv2.threshold(face_region, 125, 255, cv2.THRESH_BINARY)

        # # Replace the face region in the frame with the thresholded face region
        frame[y:y+height, x:x+width] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(
            image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # draw contours on the original image
        image_copy = frame.copy()
        # shifted_contours = tuple((u + x, v + y) for u, v in contours)
        # cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1,
        #                  color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        shifted_contours = []
        for contour in contours:
            # Shift each point of the contour by x and y
            shifted_points = tuple(
                (point[0][0] + x, point[0][1] + y) for point in contour)
            shifted_contours.append(shifted_points)

        # Display the shifted contours on the image (optional)
        for contour in shifted_contours:
            cv2.polylines(image_copy, [np.array(contour)],
                          isClosed=True, color=(0, 255, 0), thickness=2)

        # see the results
        cv2.imshow('None approximation', image_copy)

    # Display the frame with faces and thresholded face regions
    cv2.imshow("Faces with Threshold", frame)

    if (cv2.waitKey(1) == ord("q")):
        break

camera.release()
cv2.destroyAllWindows()
