import os
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd

# Rutas y variables
from Utils.paths import static_model_data_labels_path, static_model_data_path
from Utils.config import id_camera

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Cargar el archivo de etiquetas predefinido
labels_df = pd.read_csv(static_model_data_labels_path, header=None, index_col=0)
labels_dict = labels_df[1].to_dict()

# Identificador de la letra a capturar
label_id = 0  # Cambia este valor según la letra que quieras capturar (0-20)


# Función para capturar keypoints utilizando MediaPipe
def capture_keypoints(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir la imagen a RGB
    results = model.process(image_rgb)  # Procesar la imagen para detectar manos
    if results.multi_hand_landmarks:  # Si se detectan manos
        hand_landmarks = results.multi_hand_landmarks[0]  # Tomar la primera mano detectada
        keypoints = np.array(
            [[landmark.x, landmark.y] for landmark in hand_landmarks.landmark])  # Extraer los puntos clave
        # Normalizar coordenadas restando la posición de la muñeca
        wrist = keypoints[0]
        keypoints -= wrist
        keypoints = keypoints.flatten()  # Aplanar el array de puntos clave
        return keypoints, results.multi_hand_landmarks
    else:
        keypoints = np.zeros(21 * 2)  # Si no se detecta mano, devolver un array de ceros
        return keypoints, None


# Función para guardar los datos
def save_data(keypoints, label, keypoints_file):
    label = int(label)  # Convertir la etiqueta a entero
    data = np.hstack([label, keypoints])  # Concatenar la etiqueta con los puntos clave
    try:
        with open(keypoints_file, 'a') as f:
            np.savetxt(f, [data], delimiter=',',
                       fmt='%d' + ',%1.5f' * keypoints.size)  # Guardar los datos en el archivo
        print(f"Data saved for label: {label}")
    except PermissionError as e:  # Manejar el error de permiso
        print(f"Permission error: {e}")
        alternative_file = keypoints_file.replace(".csv",
                                                  "_backup.csv")  # Guardar en un archivo de respaldo si hay error
        with open(alternative_file, 'a') as f:
            np.savetxt(f, [data], delimiter=',', fmt='%d' + ',%1.5f' * keypoints.size)
        print(f"Data saved for label: {label} in backup file")


# Función para contar muestras de una etiqueta específica
def count_samples_for_label(keypoints_file, label_id):
    if os.path.exists(keypoints_file):  # Verificar si el archivo existe
        data = np.loadtxt(keypoints_file, delimiter=',')  # Cargar los datos del archivo
        if data.ndim == 1:  # Si solo hay una fila en el archivo
            labels = [int(data[0])]
        else:
            labels = data[:, 0].astype(int)  # Extraer las etiquetas de las filas
        count = list(labels).count(label_id)  # Contar las muestras para la etiqueta específica
        print(f"Label {label_id} ({labels_dict[label_id]}): {count} samples")  # Mostrar el conteo de muestras
    else:
        print("No samples captured yet.")  # Mensaje si no hay muestras capturadas


# Función principal
def main():
    print("Press 's' to start capturing data and 'q' to quit and save data.")  # Instrucciones para el usuario
    label = labels_dict[label_id]  # Obtener la etiqueta correspondiente al ID
    print(f"Capturing data for label: {label} (ID: {label_id})")  # Mensaje de inicio de captura

    cap = cv2.VideoCapture(id_camera)  # Iniciar la captura de video desde la cámara
    capturing = False  # Bandera para controlar la captura
    while cap.isOpened():
        ret, frame = cap.read()  # Leer un frame de la cámara
        if not ret:
            break

        keypoints, hand_landmarks = capture_keypoints(frame, hands)  # Capturar los puntos clave de la mano
        if hand_landmarks:  # Si se detectaron manos
            for hand_landmark in hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmark,
                                          mp_hands.HAND_CONNECTIONS)  # Dibujar las marcas de la mano

        cv2.imshow('Hand Keypoints', frame)  # Mostrar la imagen con los puntos clave

        key = cv2.waitKey(10)  # Esperar por una tecla
        if key & 0xFF == ord('s'):
            capturing = True  # Iniciar la captura de datos
            print(f"Started capturing data for label: {label} (ID: {label_id})")

        if key & 0xFF == ord('q'):
            break

        if capturing:
            save_data(keypoints, label_id, static_model_data_path)  # Guardar los datos capturados
            print(f"Captured data for label: {label} (ID: {label_id})")
            capturing = False  # Resetea la bandera de captura para evitar múltiples capturas sin presionar 's'
            count_samples_for_label(static_model_data_path, label_id)  # Muestra el conteo de múestras

    cap.release()  # Liberar la cámara
    cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV


if __name__ == "__main__":
    main()  # Llamar a la función principal
