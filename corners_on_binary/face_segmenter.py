import pathlib
import cv2
import numpy as np
from harris_corner_detector import get_good_features_to_track

cascade_path = pathlib.Path(cv2.__file__).parent.absolute(
) / "data/haarcascades/haarcascade_frontalface_default.xml"

print(cascade_path)

# make classifier

clf = cv2.CascadeClassifier(str(cascade_path))

# defining camera footage
camera = cv2.VideoCapture(0)


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
        final_image = np.copy(frame)

        # Extract the face region
        face_region = gray[y:y+height, x:x+width]

        # Apply threshold within the face region
        ret, thresh = cv2.threshold(face_region, 125, 255, cv2.THRESH_BINARY)

        # # Replace the face region in the frame with the thresholded face region
        frame[y:y+height, x:x+width] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_frame = np.zeros_like(frame_gray)
        face_frame[y:y+height, x:x +
                   width] = frame_gray[y:y + height, x:x+width]
        corners = get_good_features_to_track(face_frame)
        image_out = np.copy(final_image)
        for x, y in corners:
            x = np.round(x).astype(int)
            y = np.round(y).astype(int)
            cv2.circle(image_out, (x, y), radius=3,
                       color=(0, 255, 0, 255), thickness=-1)
        for (x, y, width, height) in faces:
            cv2.rectangle(image_out, (x, y),
                          (x+width, y+height), (0, 255, 0), 2)
        cv2.imshow("output", image_out)

    else:

        # Display the frame with faces and thresholded face regions
        cv2.imshow("output", frame)

    if (cv2.waitKey(1) == ord("q")):
        break

camera.release()
cv2.destroyAllWindows()
