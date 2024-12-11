# Importación de bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from mediapipe.python.solutions.holistic import FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
import os
import json
from typing import NamedTuple
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import (
    get_default_face_mesh_contours_style,
    get_default_pose_landmarks_style,
    get_default_hand_landmarks_style,
)


def analyze_h5_keypoints(h5_path):
    """
    Analiza los puntos clave contenidos en un archivo HDF5.

    Parámetros:
    h5_path -- Ruta al archivo HDF5.

    Funcionalidad:
    - Carga el archivo y analiza la estructura del DataFrame.
    - Verifica consistencia en el tamaño de los keypoints.
    - Proporciona estadísticas básicas para cada conjunto de puntos clave.
    - Verifica la cantidad de muestras y frames por muestra.
    - Detecta posibles valores nulos o anómalos en los datos.
    """
    LENGTH_KEYPOINTS = 1662  # Tamaño esperado de los keypoints

    print(f"\n\n[INFO] Análisis del archivo HDF5: {h5_path}")

    # Leer el archivo HDF5
    df = pd.read_hdf(h5_path, key='data')

    # Mostrar forma del DataFrame
    print(f"[INFO] DataFrame shape: {df.shape}")

    # Verificar columnas del DataFrame
    expected_columns = ['sample', 'frame', 'keypoints']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"[ERROR] Columnas faltantes: {missing_columns}")
        return
    print("[INFO] Todas las columnas necesarias están presentes.")

    # Verificar consistencia en la longitud de los keypoints
    keypoints_lengths = df['keypoints'].apply(len)
    if keypoints_lengths.nunique() == 1 and keypoints_lengths.iloc[0] == LENGTH_KEYPOINTS:
        print(f"[INFO] Todos los keypoints tienen la longitud esperada: {LENGTH_KEYPOINTS}")
    else:
        print(f"[WARNING] Longitudes de los keypoints inconsistentes. Resumen:")
        print(keypoints_lengths.value_counts())
        print(f"[WARNING] Longitudes diferentes a {LENGTH_KEYPOINTS} detectadas.")

    # Estadísticas generales de los keypoints
    keypoints_df = pd.DataFrame(df['keypoints'].tolist())
    print("[INFO] Estadísticas básicas de los keypoints:")
    print(keypoints_df.describe())

    # Verificar si hay valores nulos o anómalos
    if keypoints_df.isnull().values.any():
        print("[WARNING] Se detectaron valores nulos en los keypoints.")
    else:
        print("[INFO] No se detectaron valores nulos en los keypoints.")

    # Verificar cantidad de muestras y frames por muestra
    sample_counts = df.groupby('sample')['frame'].count()

    # Validaciones finales
    if sample_counts.min() < 15:
        print("[WARNING] Algunas muestras tienen menos de 15 frames, lo que podría afectar el entrenamiento.")
    else:
        print("[INFO] Todas las muestras tienen al menos 15 frames.")

    print("[INFO] Análisis completado. El archivo parece estar listo para el entrenamiento.")


def plot_history(model):
    """
    Genera gráficos del historial de entrenamiento de un modelo.

    Parámetros:
    history -- Objeto de historial devuelto por el entrenamiento del modelo.

    Funcionalidad:
    - Muestra la precisión y la pérdida durante el entrenamiento y la validación.
    - Proporciona una comparación de métricas entre épocas.
    """
    plt.figure(figsize=(18, 6))

    # Graficar precisión durante las épocas
    plt.subplot(1, 3, 1)
    plt.plot(model.history['accuracy'], label='Entrenamiento')
    plt.plot(model.history['val_accuracy'], label='Validación')
    plt.title('Precisión a través de las épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Graficar pérdida durante las épocas
    plt.subplot(1, 3, 2)
    plt.plot(model.history['loss'], label='Entrenamiento')
    plt.plot(model.history['val_loss'], label='Validación')
    plt.title('Pérdida a través de las épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Comparación de métricas entre entrenamiento y validación
    plt.subplot(1, 3, 3)
    epochs = range(1, len(model.history['accuracy']) + 1)
    plt.plot(epochs, model.history['accuracy'], label='Precisión - Entrenamiento', linestyle='--')
    plt.plot(epochs, model.history['val_accuracy'], label='Precisión - Validación', linestyle='-')
    plt.plot(epochs, model.history['loss'], label='Pérdida - Entrenamiento', linestyle='--')
    plt.plot(epochs, model.history['val_loss'], label='Pérdida - Validación', linestyle='-')
    plt.title('Comparación de Métricas')
    plt.xlabel('Épocas')
    plt.ylabel('Valor')
    plt.legend()

    plt.tight_layout()
    plt.show()


def mediapipe_detection(image, model):
    """
    Procesa una imagen con un modelo de MediaPipe para detección.

    Parámetros:
    image -- Imagen de entrada en formato BGR.
    model -- Modelo MediaPipe a usar para el procesamiento.

    Retorna:
    image -- Imagen procesada.
    results -- Resultados obtenidos por el modelo MediaPipe.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir la imagen a formato RGB
    image.flags.writeable = False  # Optimizar procesamiento al deshabilitar escritura
    results = model.process(image)  # Procesar la imagen con el modelo
    image.flags.writeable = True  # Habilitar escritura en la imagen nuevamente
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convertir la imagen de vuelta a formato BGR
    return image, results


def draw_keypoints(image, results):
    """
    Dibuja puntos clave en una imagen basándose en los resultados de MediaPipe.

    Parámetros:
    image -- Imagen sobre la cual se dibujarán los puntos clave.
    results -- Resultados obtenidos del modelo MediaPipe.
    """
    # Dibujar puntos clave en el rostro
    draw_landmarks(
        image,
        results.face_landmarks,
        FACEMESH_CONTOURS,
        DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    # Dibujar conexiones del cuerpo
    draw_landmarks(
        image,
        results.pose_landmarks,
        POSE_CONNECTIONS,
        DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    # Dibujar conexiones de la mano izquierda
    draw_landmarks(
        image,
        results.left_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    # Dibujar conexiones de la mano derecha
    draw_landmarks(
        image,
        results.right_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


words_text = {
    "adios": "ADIÓS",
    "bien": "BIEN",
    "buenas_noches": "BUENAS NOCHES",
    "buenas_tardes": "BUENAS TARDES",
    "buenos_dias": "BUENOS DÍAS",
    "como_estas": "COMO ESTÁS",
    "disculpa": "DISCULPA",
    "gracias": "GRACIAS",
    "hola": "HOLA",
    "mal": "MAL",
    "mas_o_menos": "MAS O MENOS",
    "me_ayudas": "ME AYUDAS",
    "por_favor": "POR FAVOR",
}
