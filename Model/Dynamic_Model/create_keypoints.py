import os
import pandas as pd
import numpy as np
import cv2
from mediapipe.python.solutions.hands import Hands

# Rutas
from Utils.paths import dynamic_model_frames_data_path, dynamic_model_converted_data_path

# Funciones
from Model.model_utils import mediapipe_detection


# Función para procesar los frames, extraer los keypoints y guarda en un archivo HDF5.
def process_and_save_keypoints(frames_path, save_path):
    # Inicializa un DataFrame vacío para almacenar los keypoints
    data = pd.DataFrame([])

    # Inicializa el modelo de MediaPipe Hands
    with Hands() as hands_model:
        for n_sample, sample_name in enumerate(os.listdir(frames_path), 1):
            sample_path = os.path.join(frames_path, sample_name)
            kp_seq = np.array([])  # Inicializa un array para almacenar la secuencia de keypoints

            for img_name in os.listdir(sample_path):
                img_path = os.path.join(sample_path, img_name)
                frame = cv2.imread(img_path)

                # Detecta manos en la imagen usando MediaPipe
                _, results = mediapipe_detection(frame, hands_model)

                # Extrae los keypoints de la mano si existen, de lo contrario retorna un arreglo de ceros
                lh = np.array([[res.x, res.y, res.z] for res in
                               results.multi_hand_landmarks[
                                   0].landmark]).flatten() if results.multi_hand_landmarks else np.zeros(21 * 3)

                # Añade los keypoints extraídos a la secuencia de keypoints
                kp_seq = np.concatenate([kp_seq, [lh]] if kp_seq.size > 0 else [[lh]])

            # Inserta la secuencia de keypoints en el DataFrame
            for frame_idx, keypoints in enumerate(kp_seq):
                data = pd.concat(
                    [data, pd.DataFrame({'sample': n_sample, 'frame': frame_idx + 1, 'keypoints': [keypoints]})])

    # Guarda los keypoints en un archivo HDF5 (.h5)
    data.to_hdf(save_path, key="data", mode="w")


if __name__ == "__main__":
    # Generar los keypoints de todas las palabras
    for word_name in os.listdir(dynamic_model_frames_data_path):
        word_path = os.path.join(dynamic_model_frames_data_path, word_name)
        hdf_path = os.path.join(dynamic_model_converted_data_path, f"{word_name}.h5")
        print(f'Creando keypoints de "{word_name}"...')
        process_and_save_keypoints(word_path, hdf_path)
        print("Keypoints creados!")
