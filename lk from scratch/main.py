import numpy as np
import cv2


from harris_corner_detector import get_good_features_to_track
from viola_jones import get_faces


def get_capture(type=0):

    if (type == 0):
        return cv2.VideoCapture(0)
    else:
        return cv2.VideoCapture("http://192.168.1.102:8080/video")


def main():

    # get video capture

    cap = get_capture(0)
    image_width = 0
    image_height = 0
    # parameters
    x0_arr = []
    y0_arr = []
    original_x0_arr = []
    original_y0_arr = []
    box_size = 30
    old_frame = ''
    old_frame_gray = ''
    velocity_vectors_array = ''
    # getting our capture length and width
    if cap.isOpened():
        image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        print(image_width, image_height)
        # filling the initial x0,y0,original x0, original y0
        ret, old_frame = cap.read()
        old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        corners = get_good_features_to_track(old_frame_gray)
        x0_arr = [x[0] for x in corners]
        y0_arr = [x[1] for x in corners]
        original_x0_arr = [x[0] for x in corners]
        original_y0_arr = [x[1] for x in corners]

        mask = np.zeros_like(old_frame)
    # main capture loop
    while 1:
        ret, frame = cap.read()

        # grey scale of current and old frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # get feature coordinates
        # run only if we found a face
        faces = get_faces(frame_gray)
        if (len(faces) != 0):

            x_begin, y_begin, width, height = faces[0]
            # make a frame with only the detected face
            face_frame = np.zeros_like(frame_gray)
            face_frame[y_begin:y_begin+height, x_begin:x_begin +
                       width] = frame_gray[y_begin:y_begin+height, x_begin:x_begin+width]
            corners = get_good_features_to_track(
                face_frame)
            x0_arr = [x[0] for x in corners]
            y0_arr = [x[1] for x in corners]

            image_out = np.copy(frame)
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
            cv2.imshow("output", frame)

        # cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


# run main
main()
