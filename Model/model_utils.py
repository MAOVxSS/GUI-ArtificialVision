# FUNCIONES AUXILIARES PARA LOS MODELOS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2


# Función para analizar los archivos h5
def analyze_h5_keypoints(h5_path):
    # Lee el archivo HDF5 con pandas
    df = pd.read_hdf(h5_path, key='data')

    # Muestra la forma del DataFrame y las primeras filas
    print(f"DataFrame shape: {df.shape}")
    print(df.head())

    # Verifica la consistencia de los key points
    consistent = df['keypoints'].apply(lambda x: len(x) == 63).all()
    if consistent:
        print("All keypoints entries have 63 elements.")
    else:
        print("Some keypoints entries do not have 63 elements.")

    # Convertir los keypoints en un DataFrame separado para análisis
    keypoints_df = pd.DataFrame(df['keypoints'].tolist())

    # Estadísticas básicas
    print("Basic statistics for keypoints:")
    print(keypoints_df.describe())


# Función para mostrar información extra sobre el entrenamiento de los modelos
def plot_history(history):
    plt.figure(figsize=(18, 6))

    # Graficar la precisión
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión a través de las épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Graficar la pérdida
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida a través de las épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Comparación de métricas
    plt.subplot(1, 3, 3)
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.plot(epochs, history.history['accuracy'], label='Precisión - Entrenamiento', linestyle='--')
    plt.plot(epochs, history.history['val_accuracy'], label='Precisión - Validación', linestyle='-')
    plt.plot(epochs, history.history['loss'], label='Pérdida - Entrenamiento', linestyle='--')
    plt.plot(epochs, history.history['val_loss'], label='Pérdida - Validación', linestyle='-')
    plt.title('Comparación de Métricas')
    plt.xlabel('Épocas')
    plt.ylabel('Valor')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Función para realizar preprocesar la imagen recortada
def preprocess_image(img_hand, img_size):
    img_white = np.ones((img_size, img_size, 3), np.uint8) * 255  # Crear un lienzo blanco

    h, w, _ = img_hand.shape
    aspect_ratio = h / w

    if aspect_ratio > 1:
        # Redimensionar si la altura es mayor que el ancho
        k = img_size / h
        new_width = math.ceil(k * w)
        resized_img = cv2.resize(img_hand, (new_width, img_size))
        width_margin = (img_size - new_width) // 2
        img_white[:, width_margin:width_margin + new_width] = resized_img
    else:
        # Redimensionar si el ancho es mayor que la altura
        k = img_size / w
        new_height = math.ceil(k * h)
        resized_img = cv2.resize(img_hand, (img_size, new_height))
        height_margin = (img_size - new_height) // 2
        img_white[height_margin:height_margin + new_height, :] = resized_img

    return img_white


# Función para binarizar la imagen entrante
def binarize_img(original_img, hsv_image):
    # Se recorre todos los píxeles de la imagen
    img_bin = np.zeros((original_img.shape[0], original_img.shape[1]))
    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            if (hsv_image[i, j, 0] > 107 and hsv_image[i, j, 0] < 127) and \
                    (hsv_image[i, j, 1] > 48 and hsv_image[i, j, 1] < 157) and \
                    (hsv_image[i, j, 2] > 56 and hsv_image[i, j, 2] < 186):
                img_bin[i, j] = 255
    return img_bin


# Función para procesar la imagen con MediaPipe y detectar manos
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convierte la imagen de BGR a RGB
    image.flags.writeable = False  # Optimiza la velocidad de procesamiento
    results = model.process(image)  # Procesa la imagen con el modelo de MediaPipe
    image.flags.writeable = True  # Vuelve a permitir escritura en la imagen
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convierte la imagen de vuelta a BGR
    return image, results  # Devuelve la imagen procesada y los resultados
