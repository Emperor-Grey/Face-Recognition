import cv2
import numpy as np
from keras.models import load_model

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


def load_labels(file_path):
    """Load class labels from a file."""
    with open(file_path, "r") as file:
        labels = file.readlines()
    return labels


def preprocess_image(image):
    """Preprocess the image to fit the model input requirements."""
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1  # Normalize image
    return image


def predict_class(model, image, class_names):
    """Predict the class of the image using the model."""
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score


def main():
    # Load the model and labels
    model = load_model("keras_Model.h5", compile=False)
    class_names = load_labels("labels.txt")

    # Initialize the camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            ret, image = camera.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Show the image in a window with a larger size
            display_image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
            cv2.imshow("Webcam Image", display_image)

            # Preprocess the image and make predictions
            preprocessed_image = preprocess_image(image)
            class_name, confidence_score = predict_class(model, preprocessed_image, class_names)

            # Print prediction and confidence score
            print(f"Class: {class_name}, Confidence Score: {confidence_score * 100:.2f}%")

            # Listen to the keyboard for presses, exit on 'Esc' key
            if cv2.waitKey(1) == 27:
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
