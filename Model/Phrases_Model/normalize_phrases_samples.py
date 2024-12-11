import cv2
import numpy as np
import os
import shutil
import re
from Utils.paths import phrases_model_frames_data_path, phrases_model_test_frames_data_path


def read_frames_from_directory(directory):
    """
    Lee y retorna una lista de frames desde un directorio, asegurando el orden correcto.
    """
    # Obtener lista de archivos .jpg
    filenames = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    # Ordenar los archivos numéricamente
    def extract_number(filename):
        # Extrae el número del nombre del archivo
        basename = os.path.splitext(filename)[0]
        # Remover cualquier carácter que no sea dígito
        number = ''.join(filter(str.isdigit, basename))
        return int(number) if number.isdigit() else 0

    filenames.sort(key=extract_number)

    frames = [cv2.imread(os.path.join(directory, filename)) for filename in filenames]
    return frames


def pad_frames(frames, target_frame_count=15):
    """
    Ajusta el número de frames a `target_frame_count` mediante truncamiento o relleno.
    """
    current_frame_count = len(frames)
    if current_frame_count >= target_frame_count:
        # Trunca la secuencia a los primeros `target_frame_count` frames
        return frames[:target_frame_count]
    else:
        # Repite el último frame hasta alcanzar el número deseado
        last_frame = frames[-1]
        frames.extend([last_frame.copy() for _ in range(target_frame_count - current_frame_count)])
        return frames


def clear_directory(directory):
    """
    Limpia el contenido de un directorio.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def save_normalized_frames(directory, frames):
    """
    Guarda frames normalizados en un directorio.
    """
    for i, frame in enumerate(frames, start=1):
        # Guarda el frame con un nombre numerado
        cv2.imwrite(
            os.path.join(directory, f'frame_{i}.jpg'),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 50]
        )


def process_directory(word_directory, target_frame_count=15):
    """
    Procesa un directorio de muestras y normaliza los frames de cada muestra.
    """
    for sample_name in os.listdir(word_directory):
        sample_directory = os.path.join(word_directory, sample_name)
        if os.path.isdir(sample_directory):
            # Lee frames desde el directorio de la muestra
            frames = read_frames_from_directory(sample_directory)

            # Verifica que se hayan leído frames correctamente
            if not frames:
                print(f"[WARNING] No se encontraron frames en {sample_directory}. Saltando...")
                continue

            # Normaliza los frames al tamaño objetivo
            normalized_frames = pad_frames(frames, target_frame_count)

            # Limpia el directorio actual
            clear_directory(sample_directory)

            # Guarda los frames normalizados
            save_normalized_frames(sample_directory, normalized_frames)


if __name__ == "__main__":

    # Lista todas las palabras en el directorio de frames
    # Si estás procesando los datos de entrenamiento:
    # word_ids = [word for word in os.listdir(phrases_model_frames_data_path)]

    word_ids = ["k_der", "k_izq"]

    # Si estás procesando los datos de prueba:
    # word_ids = [word for word in os.listdir(phrases_model_test_frames_data_path)]

    # Frames máximos para cada acción
    max_actions_frames = 15

    # Procesa cada palabra en el directorio
    for word_id in word_ids:

        # Para los datos de entrenamiento:
        word_path = os.path.join(phrases_model_frames_data_path, word_id)

        # Para los datos de prueba:
        # word_path = os.path.join(phrases_model_test_frames_data_path, word_id)

        if os.path.isdir(word_path):
            print(f'Normalizando frames para "{word_id}"...')

            # Comienza el normalizado
            process_directory(word_path, max_actions_frames)
