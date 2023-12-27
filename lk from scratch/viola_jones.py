import pathlib
import cv2


def get_faces(frame_gray):

    cascade_path = pathlib.Path(cv2.__file__).parent.absolute(
    ) / "data/haarcascades/haarcascade_frontalface_default.xml"

    # make classifier

    clf = cv2.CascadeClassifier(str(cascade_path))

    faces = clf.detectMultiScale(
        frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

    # rendering the faces
    # for (x, y, width, height) in faces:
    #     cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
    #     frame[y:y+height, x:x+width, 0] = gray[y:y+height, x:x+width]
    #     frame[y:y+height, x:x+width, 1] = gray[y:y+height, x:x+width]
    #     frame[y:y+height, x:x+width, 2] = gray[y:y+height, x:x+width]
