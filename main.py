import json
import os
import joblib
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from skimage.io import imread
from skimage.transform import resize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense, Dropout, Conv2D
import tensorflow as tf
from onnxmltools import convert_sklearn, convert_keras
from onnxconverter_common import FloatTensorType

# Model names and file paths
svm_model_name = "svm_model"
rf_model_name = "randomforest_model"
cnn_model_name = "cnn_model"
ensemble_model_name = "ensemble_model"
models_dir = "learned_models"  # Only the subdirectory name


def save_model(model, model_name, n_support=None, input_shape=None, target_opset=11):
    os.makedirs(models_dir, exist_ok=True)  # Create the directory if it doesn't exist
    model_path = os.path.join(models_dir, model_name)

    if isinstance(model, SVC) or isinstance(model, RandomForestClassifier):
        if n_support is not None:
            try:
                model_onnx = convert_sklearn(model, target_opset=target_opset, initial_types=[('input', FloatTensorType([None, n_support]))])
                onnx_path = model_path + ".onnx"
                with open(onnx_path, "wb") as f:
                    f.write(model_onnx.SerializeToString())
                print(f"{model_name} saved to {onnx_path}")
                return
            except Exception as e:
                print(f"Failed to convert to ONNX: {e}")
    elif isinstance(model, tf.keras.Model):
        if input_shape is not None:
            try:
                model_onnx = convert_keras(model, model.name, target_opset=9,
                                           initial_types=[('input', FloatTensorType(input_shape))])
                onnx_path = model_path + ".onnx"
                with open(onnx_path, "wb") as f:
                    f.write(model_onnx.SerializeToString())
                print(f"{model_name} saved to {onnx_path}")
                return
            except Exception as e:
                print(f"Failed to convert to ONNX: {e}")

    try:
        # If ONNX conversion fails, save as joblib
        joblib_path = model_path + ".joblib"
        joblib.dump(model, joblib_path)
        print(f"{model_name} saved to {joblib_path}")
    except Exception as e:
        print(f"Failed to save {model_name} model: {e}")


# Set the path to the directory containing the labeled face images
data_dir = "output_faces"

# Get a list of subdirectories (each corresponds to a person's images)
subdirs = [os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir)]

X = []  # Features (images)
y = []  # Labels (person's ID)

# Create an empty label-to-person mapping dictionary
label_to_person = {}

for person_id, person_dir in enumerate(subdirs):
    person_name = os.path.basename(person_dir)  # Get the person's name from the directory name
    label_to_person[person_id] = person_name  # Add the label-to-person mapping
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        img = imread(img_path, as_gray=True)  # Read image in grayscale
        img = resize(img, (100, 100))  # Resize image to a common size
        X.append(img)  # Keep images in original 2D format
        y.append(person_id)

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model 1: Support Vector Machine (SVM)
svm_model = SVC()
svm_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
svm_pred = svm_model.predict(X_test.reshape(X_test.shape[0], -1))
svm_accuracy = accuracy_score(y_test, svm_pred)
print("SVM Accuracy:", svm_accuracy)

# Get the number of support vectors
n_support_svm = svm_model.n_support_[0]

# Save SVM model to ONNX format
save_model(svm_model, svm_model_name, n_support=n_support_svm)

# Model 2: Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
rf_pred = rf_model.predict(X_test.reshape(X_test.shape[0], -1))
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)

# Save Random Forest model to ONNX format with a compatible target_opset
save_model(rf_model, rf_model_name, target_opset=11)

# Define the input dimensions based on your model's input shape.
batchSize = 1       # Adjust based on your model's batch size
channels = 1        # Number of channels (1 for grayscale)
inputHeight = 100   # Model's input height
inputWidth = 100    # Model's input width

# Model 3: CNN model using Conv2D layers directly
cnn_model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(inputHeight, inputWidth, channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(subdirs), activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train.reshape(X_train.shape[0], inputHeight, inputWidth, channels), y_train, epochs=10, batch_size=batchSize)

# print("x_train shape is: ",X_train.shape[0])

# Evaluate the CNN model on the test set
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(X_test.reshape(X_test.shape[0], inputHeight, inputWidth, channels), y_test)
print("CNN Test Accuracy:", cnn_test_accuracy)

# Save CNN model to ONNX formats
save_model(cnn_model, cnn_model_name, input_shape=(batchSize, inputHeight, inputWidth, channels))


# Custom Keras Classifier
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, keras_model):
        self.keras_model = keras_model

    def fit(self, X_train_fit, y_train_fit):
        # Your fit logic here
        X_train_fit_reshaped = X_train_fit.reshape(X_train_fit.shape[0], 100, 100, 1)
        self.keras_model.fit(X_train_fit_reshaped, y_train_fit, epochs=10, batch_size=32)
        return self

    def predict(self, X_predict):
        # Your predict logic here
        X_predict_reshaped = X_predict.reshape(X_predict.shape[0], 100, 100, 1)
        return self.keras_model.predict(X_predict_reshaped)

    def predict_proba(self, X_predict):
        # Implement predict_proba to return class probabilities
        X_predict_reshaped = X_predict.reshape(X_predict.shape[0], 100, 100, 1)
        predictions = self.keras_model.predict(X_predict_reshaped)
        class_probs = np.zeros_like(predictions)
        class_probs[np.arange(len(predictions)), predictions.argmax(1)] = 1
        return class_probs

    def get_config(self):
        return self.keras_model.get_config()


# Create custom classifiers for each model
svm_classifier = SVC(probability=True)
rf_classifier = RandomForestClassifier()
cnn_classifier = KerasClassifierWrapper(cnn_model)

# Create the ensemble of classifiers
ensemble_model = VotingClassifier(estimators=[
    ('svm', svm_classifier),
    ('rf', rf_classifier),
    ('cnn', cnn_classifier)
], voting='soft')

# Fit the ensemble model
ensemble_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Predict using the ensemble model
ensemble_pred = ensemble_model.predict(X_test.reshape(X_test.shape[0], -1))

# Calculate the accuracy of the ensemble model
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
print("Ensemble Model Accuracy:", ensemble_accuracy)

# Save the ensemble model to ONNX format
save_model(ensemble_model, ensemble_model_name)

# Save the label-to-person mappings to a JSON file with non-ASCII characters
with open('label_to_person.json', 'w', encoding='utf-8') as file:
    json.dump(label_to_person, file, ensure_ascii=False)


def predict_person(image_path, svm_model_predict, rf_model_predict, cnn_model_predict, ensemble_model_predict, subdirs_predict):
    test_image = imread(image_path, as_gray=True)
    test_image = resize(test_image, (100, 100))

    # Reshape the image for the models
    test_image_reshaped = test_image.reshape(1, -1)  # SVM and Random Forest
    test_image_cnn_reshaped = test_image.reshape(1, 100, 100, 1)  # CNN

    # SVM prediction
    svm_prediction = svm_model_predict.predict(test_image_reshaped)
    svm_confidence = np.max(svm_model_predict.decision_function(test_image_reshaped))
    svm_predicted_person_name = os.path.basename(subdirs_predict[svm_prediction[0]])

    # Random Forest prediction
    rf_prediction = rf_model_predict.predict(test_image_reshaped)
    rf_confidence = np.max(rf_model_predict.predict_proba(test_image_reshaped))
    rf_predicted_person_name = os.path.basename(subdirs_predict[rf_prediction[0]])

    # CNN prediction
    cnn_prediction = cnn_model_predict.predict(test_image_cnn_reshaped)
    cnn_confidence = np.max(cnn_prediction, axis=1)  # Get the maximum confidence value
    cnn_predicted_person_id = np.argmax(cnn_prediction)
    cnn_predicted_person_name = os.path.basename(subdirs_predict[cnn_predicted_person_id])

    # Ensemble prediction
    ensemble_predictions = ensemble_model_predict.predict_proba(test_image_reshaped)
    ensemble_confidence = np.max(ensemble_predictions)
    ensemble_predicted_person_id = np.argmax(ensemble_predictions)
    ensemble_predicted_person_name = os.path.basename(subdirs_predict[ensemble_predicted_person_id])

    return (svm_predicted_person_name, svm_confidence), (rf_predicted_person_name, rf_confidence), (cnn_predicted_person_name, cnn_confidence), (ensemble_predicted_person_name, ensemble_confidence)


class PhotoPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Prediction App")

        self.image_path = None
        self.image = None
        self.photo = None

        self.image_label = tk.Label(root, text="Select an image")
        self.image_label.grid(row=0, column=0, columnspan=2)

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.grid(row=1, column=0, columnspan=2)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.grid(row=2, column=0, columnspan=2)

        self.svm_result = tk.Label(root, text="")
        self.svm_result.grid(row=3, column=0, columnspan=2)

        self.rf_result = tk.Label(root, text="")
        self.rf_result.grid(row=4, column=0, columnspan=2)

        self.cnn_result = tk.Label(root, text="")
        self.cnn_result.grid(row=5, column=0, columnspan=2)

        self.ensemble_result = tk.Label(root, text="")
        self.ensemble_result.grid(row=6, column=0, columnspan=2)

        self.svm_confidence_label = tk.Label(root, text="")
        self.svm_confidence_label.grid(row=7, column=0)

        self.rf_confidence_label = tk.Label(root, text="")
        self.rf_confidence_label.grid(row=7, column=1)

        self.cnn_confidence_label = tk.Label(root, text="")
        self.cnn_confidence_label.grid(row=8, column=0)

        self.ensemble_confidence_label = tk.Label(root, text="")
        self.ensemble_confidence_label.grid(row=8, column=1)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            self.image = Image.open(self.image_path)
            self.image.thumbnail((300, 300))
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)

    def predict(self):
        if hasattr(self, 'image_path'):
            svm_result, rf_result, cnn_result, ensemble_result = predict_person(self.image_path, svm_model, rf_model, cnn_model, ensemble_model, subdirs)

            # Load and preprocess the image for the ensemble model
            test_image = imread(self.image_path, as_gray=True)
            test_image = resize(test_image, (100, 100))
            test_image_reshaped = test_image.reshape(1, -1)

            # Predict using the ensemble model
            ensemble_confidence = np.max(ensemble_model.predict_proba(test_image_reshaped))

            self.svm_result.config(text="SVM Predicted Person: " + svm_result[0])
            self.rf_result.config(text="Random Forest Predicted Person: " + rf_result[0])
            self.cnn_result.config(text="CNN Predicted Person: " + cnn_result[0])
            self.ensemble_result.config(
                text="Ensemble Predicted Person: " + ensemble_result[0])  # Display ensemble prediction

            self.svm_confidence_label.config(text="SVM Confidence: {:.2f}".format(svm_result[1]))
            self.rf_confidence_label.config(text="RF Confidence: {:.2f}".format(rf_result[1]))
            self.cnn_confidence_label.config(text="CNN Confidence: {:.2f}".format(cnn_result[1].item()))
            self.ensemble_confidence_label.config(text="Ensemble Confidence: {:.2f}".format(ensemble_confidence.item()))
        else:
            self.svm_result.config(text="No image selected.")
            self.rf_result.config(text="")
            self.cnn_result.config(text="")
            self.ensemble_result.config(text="")
            self.svm_confidence_label.config(text="")
            self.rf_confidence_label.config(text="")
            self.cnn_confidence_label.config(text="")
            self.ensemble_confidence_label.config(text="")


def main():
    root = tk.Tk()
    app = PhotoPredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
