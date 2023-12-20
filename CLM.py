import dlib
import cv2


predictor_path = './shape_predictor/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

camera = cv2.VideoCapture(0)


while True:
    

    _, frame = camera.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    detector = dlib.get_frontal_face_detector()

    # Detect faces
    faces = detector(gray)

    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(gray, face)

        # Draw facial landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Display the result
    cv2.imshow('Facial Landmarks', frame)

    if (cv2.waitKey(1) == ord("q")):
            break

camera.release()
cv2.destroyAllWindows()