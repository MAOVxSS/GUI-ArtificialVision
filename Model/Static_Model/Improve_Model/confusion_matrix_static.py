import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from Utils.paths import generated_models_path, static_model_test_data_path, static_model_data_labels_path
from Utils.config import static_model_name

def load_data_from_csv(csv_path):
    """
    Loads sequences and labels from a CSV file.

    Parameters:
    - csv_path: Path to the CSV file containing the data.

    Returns:
    - X: Numpy array of keypoints sequences.
    - y: Numpy array of labels.
    """
    data = pd.read_csv(csv_path, header=None)
    labels = data.iloc[:, 0].values  # First column contains the labels
    sequences = data.iloc[:, 1:].values  # All other columns are keypoints

    return np.array(sequences), np.array(labels)

def map_labels_to_classes(labels_path):
    """
    Maps label indices to their corresponding class names from a CSV file.

    Parameters:
    - labels_path: Path to the CSV file containing the label-class mapping.

    Returns:
    - class_names: List of class names corresponding to the labels.
    """
    labels_df = pd.read_csv(labels_path, header=None, index_col=0)
    class_names = labels_df[1].to_list()
    return class_names

def generate_confusion_matrix_from_csv(csv_path, model_path, labels_path):
    """
    Generates and displays the confusion matrix using data from a CSV file.

    Parameters:
    - csv_path: Path to the CSV file containing the data.
    - model_path: Path to the trained model file.
    - labels_path: Path to the CSV file containing the label-class mapping.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV file not found at: {csv_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Trained model not found at: {model_path}")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"[ERROR] Labels file not found at: {labels_path}")

    # Load the data
    X, y_true = load_data_from_csv(csv_path)

    # Normalize the keypoints
    X = X - X.mean(axis=1, keepdims=True)

    # Load the class names
    class_names = map_labels_to_classes(labels_path)

    # Load the trained model
    model = load_model(model_path)
    print("[INFO] Model loaded successfully.")

    # Make predictions
    y_pred_prob = model.predict(X)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate detailed metrics
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("[INFO] Classification Report:")
    print(report)

    # Visualize the confusion matrix
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(include_values=True, cmap='Blues', ax=plt.gca(), xticks_rotation='vertical')
    plt.title('Confusion Matrix from CSV Data')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    CSV_PATH = os.path.join(static_model_test_data_path)  # Path to the CSV file
    MODEL_PATH = os.path.join(generated_models_path, static_model_name)  # Path to the trained model
    LABELS_PATH = os.path.join(static_model_data_labels_path)  # Path to the label-class mapping

    generate_confusion_matrix_from_csv(CSV_PATH, MODEL_PATH, LABELS_PATH)
