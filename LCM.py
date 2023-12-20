import dlib
import cv2

predictor_path = '/shape_predictor/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

img = cv2.imread('your_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = detector(gray)

for face in faces:
    # Predict facial landmarks
    landmarks = predictor(gray, face)

    # Draw facial landmarks
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

# Display the result
cv2.imshow('Facial Landmarks', img)
cv2.waitKey(0)
# cv2.destroyAllWindows()