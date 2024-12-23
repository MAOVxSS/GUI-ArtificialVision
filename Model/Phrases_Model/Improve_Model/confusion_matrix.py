import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Utils.paths import phrases_model_json_data_path, phrases_model_test_converted_data_path, generated_models_path, \
    phrases_model_keras_path
from Utils.config import phrases_model_keras_name

def load_test_data(word_ids, max_length_frames=15):
    """
    Carga las secuencias de prueba y sus etiquetas desde los archivos HDF5.

    Parámetros:
    - word_ids: Lista de identificadores de palabras (clases).
    - max_length_frames: Longitud máxima de las secuencias.

    Retorna:
    - X_test: Numpy array con las secuencias de prueba.
    - y_test: Numpy array con las etiquetas de prueba.
    """
    sequences, labels = [], []
    for word_index, word_id in enumerate(word_ids):
        hdf_path = os.path.join(phrases_model_test_converted_data_path, f"{word_id}_test.h5")
        if not os.path.exists(hdf_path):
            print(f"[WARNING] Archivo HDF5 de prueba no encontrado: {hdf_path}. Saltando...")
            continue
        data = pd.read_hdf(hdf_path, key='data')
        for _, df_sample in data.groupby('sample'):
            seq_keypoints = np.stack(df_sample['keypoints'].values)
            sequences.append(seq_keypoints)
            labels.append(word_index)

    if not sequences or not labels:
        raise ValueError("[ERROR] No se encontraron secuencias ni etiquetas para generar la matriz de confusión.")

    # Convertir las secuencias a numpy arrays
    sequences = np.array(sequences)

    # Aplanar los keypoints por frame para normalización
    num_samples, seq_len, num_keypoints = sequences.shape
    sequences_flat = sequences.reshape(-1, num_keypoints)

    # Normalizar los keypoints usando el mismo scaler que en el entrenamiento
    # Aquí asumimos que usaste StandardScaler y lo guardaste; si no, debes guardarlo durante el entrenamiento.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    sequences_flat = scaler.fit_transform(sequences_flat)

    # Restaurar la forma original
    sequences = sequences_flat.reshape(num_samples, seq_len, num_keypoints)

    # Ajustar las secuencias al tamaño máximo definido
    sequences = pad_sequences(
        sequences, maxlen=max_length_frames, padding='post', truncating='post', dtype='float32'
    )

    X_test = np.array(sequences)
    y_test = np.array(labels)

    return X_test, y_test

def generate_confusion_matrix():
    """
    Genera y muestra la matriz de confusión para el modelo de reconocimiento de señas.
    """
    # Cargar los identificadores de palabras
    with open(phrases_model_json_data_path, 'r') as json_file:
        data = json.load(json_file)
        word_ids = data.get('word_ids', [])
        if not word_ids:
            raise ValueError("[ERROR] No se encontraron identificadores de palabras en el archivo JSON.")

    # Cargar el modelo entrenado
    model_path = os.path.join(phrases_model_keras_path, phrases_model_keras_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Modelo entrenado no encontrado en la ruta: {model_path}")

    model = load_model(model_path)
    print("[INFO] Modelo cargado exitosamente.")

    # Cargar los datos de prueba
    X_test, y_test = load_test_data(word_ids)

    # Realizar predicciones
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Generar la matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calcular métricas detalladas
    report = classification_report(y_test, y_pred, target_names=word_ids)
    print("[INFO] Reporte de clasificación:")
    print(report)

    # Visualizar la matriz de confusión
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=word_ids)
    disp.plot(include_values=True, cmap='Blues', ax=plt.gca(), xticks_rotation='vertical')
    plt.title('Confusion Matrix of the Sign Recognition Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_confusion_matrix()
