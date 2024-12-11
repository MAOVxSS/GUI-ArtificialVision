import os
import cv2
import numpy as np
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from Utils.paths import (phrases_model_frames_data_path, phrases_model_converted_data_path,
                         phrases_model_test_frames_data_path, phrases_model_test_converted_data_path)
from Model.model_utils import analyze_h5_keypoints

def create_keypoints(word_id, words_path, h5_path):
    """
    Genera y guarda los keypoints de una palabra en un archivo HDF5.

    Parámetros:
    word_id -- Identificador único de la palabra (nombre del directorio de frames).
    words_path -- Ruta donde se encuentran los directorios de las palabras.
    hdf_path -- Ruta donde se almacenará el archivo HDF5 generado.

    Funcionalidad:
    - Itera sobre los frames de cada muestra de la palabra.
    - Calcula los keypoints de las muestras usando MediaPipe Holistic.
    - Almacena los keypoints en un DataFrame y los guarda en formato HDF5.
    - Al finalizar, analiza el archivo HDF5 y muestra estadísticas relevantes.

    Retorna:
    - Nada, pero guarda los keypoints procesados en `hdf_path`.
    """
    # Inicializa un DataFrame vacío para almacenar los keypoints
    data = pd.DataFrame([])
    frames_path = os.path.join(words_path, word_id)

    # Asegura que la carpeta de keypoints existe
    if not os.path.exists(phrases_model_converted_data_path):
        os.makedirs(phrases_model_converted_data_path)

    # Inicializa el modelo Holistic para procesar los frames
    with Holistic() as holistic:
        print(f'Creando keypoints de "{word_id}"...')

        # Obtiene la lista de muestras y el conteo total
        sample_list = os.listdir(frames_path)
        sample_count = len(sample_list)

        # Itera sobre cada muestra de la palabra
        for n_sample, sample_name in enumerate(sample_list, start=1):
            sample_path = os.path.join(frames_path, sample_name)

            # Obtiene los keypoints para la muestra actual
            kp_seq = []
            for img_name in os.listdir(sample_path):
                img_path = os.path.join(sample_path, img_name)
                frame = cv2.imread(img_path)

                # Procesa el frame para obtener keypoints con MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                # Extrae los keypoints del frame
                pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                                 results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
                face = np.array([[res.x, res.y, res.z] for res in
                                 results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
                left_hand = np.array([[res.x, res.y, res.z] for res in
                                      results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
                right_hand = np.array([[res.x, res.y, res.z] for res in
                                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

                kp_seq.append(np.concatenate([pose, face, left_hand, right_hand]))

            # Agrega la secuencia de keypoints al DataFrame
            for frame_idx, keypoints in enumerate(kp_seq, start=1):
                data = pd.concat([data, pd.DataFrame({
                    'sample': n_sample,
                    'frame': frame_idx,
                    'keypoints': [keypoints]
                })])

            # Progreso en consola para seguimiento
            print(f"{n_sample}/{sample_count}", end="\r")

    # Guarda los keypoints en un archivo HDF5
    data.to_hdf(h5_path, key="data", mode="w")
    print(f"Keypoints creados! ({sample_count} muestras)")

    # Analiza el archivo HDF5 recién creado
    analyze_h5_keypoints(h5_path)


if __name__ == "__main__":
    """
    Procesa todas las palabras o un conjunto específico y genera los keypoints.
    """

    # Generar keypoints para todas las palabras normales
    # word_ids = [word for word in os.listdir(phrases_model_frames_data_path)]

    # Generar keypoints para todas las palabras de pruebas
    # word_ids = [word for word in os.listdir(phrases_model_test_frames_data_path)]

    # Alternativamente, generar keypoints solo para palabras seleccionadas
    word_ids = ["k_izq", "k_der"]

    # Procesa cada palabra en la lista
    for word_id in word_ids:

        #Guardar archivos normales
        h5_path = os.path.join(phrases_model_converted_data_path, f"{word_id}.h5")

        # Guardar archivos de prueba
        # h5_path = os.path.join(phrases_model_test_converted_data_path, f"{word_id}.h5")

        # Guardar para archivos normales
        create_keypoints(word_id, phrases_model_frames_data_path, h5_path)

        # Guardar para archivos de prueba
        # create_keypoints(word_id, phrases_model_test_frames_data_path, h5_path)
