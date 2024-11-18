import cv2
import numpy as np
import os
import shutil
from Utils.paths import phrases_model_frames_data_path


def read_frames_from_directory(directory):
    """
    Lee y retorna una lista de frames desde un directorio.

    Parámetros:
    directory -- Ruta del directorio donde se encuentran los frames.

    Funcionalidad:
    - Filtra y ordena los archivos del directorio con extensión `.jpg`.
    - Carga cada archivo como un frame usando OpenCV.

    Retorna:
    - Una lista de frames en formato numpy array.
    """
    return [
        cv2.imread(os.path.join(directory, filename))
        for filename in sorted(os.listdir(directory))
        if filename.endswith('.jpg')
    ]


def interpolate_frames(frames, target_frame_count=15):
    """
    Interpola frames para igualar la cantidad de frames a un valor objetivo.

    Parámetros:
    frames -- Lista de frames actuales.
    target_frame_count -- Número objetivo de frames.

    Funcionalidad:
    - Calcula índices equidistantes entre frames existentes.
    - Genera frames interpolados combinando frames vecinos usando ponderación.

    Retorna:
    - Una lista de frames interpolados con el tamaño `target_frame_count`.
    """
    current_frame_count = len(frames)
    if current_frame_count == target_frame_count:
        return frames

    # Genera índices interpolados de manera uniforme
    indices = np.linspace(0, current_frame_count - 1, target_frame_count)
    interpolated_frames = []
    for i in indices:
        # Encuentra los índices de los frames vecinos
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx

        # Genera un frame interpolado ponderado entre vecinos
        interpolated_frame = cv2.addWeighted(frames[lower_idx], 1 - weight, frames[upper_idx], weight, 0)
        interpolated_frames.append(interpolated_frame)

    return interpolated_frames


def normalize_frames(frames, target_frame_count=15):
    """
    Normaliza el número de frames a una cantidad específica.

    Parámetros:
    frames -- Lista de frames actuales.
    target_frame_count -- Número objetivo de frames.

    Funcionalidad:
    - Si hay menos frames, interpola para alcanzar el número objetivo.
    - Si hay más frames, selecciona de manera equidistante.
    - Si la cantidad es igual al objetivo, devuelve los frames sin cambios.

    Retorna:
    - Una lista de frames normalizados con tamaño `target_frame_count`.
    """
    current_frame_count = len(frames)
    if current_frame_count < target_frame_count:
        # Caso de menos frames: interpolación
        return interpolate_frames(frames, target_frame_count)
    elif current_frame_count > target_frame_count:
        # Caso de exceso de frames: selección equidistante
        step = current_frame_count / target_frame_count
        indices = np.arange(0, current_frame_count, step).astype(int)[:target_frame_count]
        return [frames[i] for i in indices]
    else:
        # Caso exacto: devuelve los frames sin modificaciones
        return frames


def clear_directory(directory):
    """
    Limpia el contenido de un directorio.

    Parámetros:
    directory -- Ruta del directorio a limpiar.

    Funcionalidad:
    - Elimina todos los archivos y subdirectorios en el directorio especificado.
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

    Parámetros:
    directory -- Directorio donde se guardarán los frames.
    frames -- Lista de frames a guardar.

    Funcionalidad:
    - Itera sobre cada frame y lo guarda en el directorio.
    - Configura la calidad de los frames guardados a un valor comprimido.
    """
    for i, frame in enumerate(frames, start=1):
        # Guarda el frame con un nombre numerado
        cv2.imwrite(
            os.path.join(directory, f'frame_{i:02}.jpg'),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 50]
        )


def process_directory(word_directory, target_frame_count=15):
    """
    Procesa un directorio de muestras y normaliza los frames de cada muestra.

    Parámetros:
    word_directory -- Ruta del directorio que contiene las muestras.
    target_frame_count -- Número objetivo de frames por muestra.

    Funcionalidad:
    - Itera sobre cada subdirectorio de muestras.
    - Lee los frames de cada muestra y los normaliza.
    - Limpia el directorio y guarda los frames normalizados.
    """
    for sample_name in os.listdir(word_directory):
        sample_directory = os.path.join(word_directory, sample_name)
        if os.path.isdir(sample_directory):
            # Lee frames desde el directorio de la muestra
            frames = read_frames_from_directory(sample_directory)

            # Normaliza los frames al tamaño objetivo
            normalized_frames = normalize_frames(frames, target_frame_count)

            # Limpia el directorio actual
            clear_directory(sample_directory)

            # Guarda los frames normalizados
            save_normalized_frames(sample_directory, normalized_frames)


if __name__ == "__main__":
    # Lista todas las palabras en el directorio de acciones de frames
    # word_ids = [word for word in os.listdir(phrases_model_frames_data_path)]

    # Alternativamente, normalizar solo las palabras seleccionadas
    # word_ids = ["bien", ...]
    word_ids = ["adios", "cuidate"]

    # Frames máximos para cada acción
    max_actions_frames = 15

    # Procesa cada palabra en el directorio
    for word_id in word_ids:
        word_path = os.path.join(phrases_model_frames_data_path, word_id)
        if os.path.isdir(word_path):
            print(f'Normalizando frames para "{word_id}"...')

            # Comienza el normalizado
            process_directory(word_path, max_actions_frames)
