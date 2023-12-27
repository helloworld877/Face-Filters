import dlib 
import numpy as np
import skimage as io

def fine_tuning(image_paths, landmark_paths):
    images = []
    landmarks = []
    for image_path, landmark_path in zip(image_paths, landmark_paths):
        image = io.imread(image_path)
        landmarks_file = np.loadtxt(landmark_path)
        images.append(image)
        landmarks.append(landmarks_file)
    images = np.array(images) / 255.0

    predictor = dlib.shape_predictor("./shape_predictor/shape_predictor_68_face_landmarks.dat")

    optimizer = dlib.adam_optimizer()

    def mean_squared_error(predicted_landmarks, true_landmarks):
        error = np.mean(np.square(predicted_landmarks - true_landmarks))
        return error

    for image, true_landmarks in zip(images, landmarks):
        # Predict landmarks using the current model
        predicted_landmarks = predictor(image, dlib.rectangle(0, 0, image.shape[1], image.shape[0]))

        # Calculate the loss
        loss = mean_squared_error(predicted_landmarks, true_landmarks)

        # Update model parameters using the optimizer
        optimizer.update(loss)
