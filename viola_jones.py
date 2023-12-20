import pathlib
import cv2

cascade_path = "./data/haarcascades/haarcascade_frontalface_default.xml"

# print(cascade_path)

# make classifier

clf = cv2.CascadeClassifier(str(cascade_path))

# defining camera footage
camera = cv2.VideoCapture(0)


# render loop

while True:
    # reading form camera and detecting faces
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # rendering the faces
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        frame[y:y+height, x:x+width, 0] = gray[y:y+height, x:x+width]
        frame[y:y+height, x:x+width, 1] = gray[y:y+height, x:x+width]
        frame[y:y+height, x:x+width, 2] = gray[y:y+height, x:x+width]

    cv2.imshow("faces", frame)

    if (cv2.waitKey(1) == ord("q")):
        break

camera.release()
cv2.destroyAllWindows()
