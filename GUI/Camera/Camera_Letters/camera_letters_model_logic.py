from tkinter import PhotoImage, Label
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
from pathlib import Path
import pandas as pd

# Rutas a los archivos JSON con información de las letras y tips
from Utils.paths import assets_json_letters_info_path

# Importar las rutas de los recursos desde módulos de utilidades
from Utils.paths import (static_model_data_labels_path, generated_models_path,
                         assets_camera_path)
from Utils.config import static_model_lite_name, id_camera


# Funciones para generar la ruta completa hacia los archivos de recursos (imágenes)
def relative_to_assets_camera(path: str) -> Path:
    return assets_camera_path / Path(path)


# Inicializar MediaPipe Hands para detectar las manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Cargar el modelo tflite para reconocimiento de señas
interpreter = tf.lite.Interpreter(model_path=os.path.join(generated_models_path, static_model_lite_name))
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cargar el archivo de etiquetas predefinido (para identificar señas)
labels_df = pd.read_csv(static_model_data_labels_path, header=None, index_col=0)
labels_dict = labels_df[1].to_dict()

# Variables globales para la captura de video
cap = None
video_update_id = None


# Función para predecir la seña basada en los key points de la mano
def predict_keypoints(keypoints):
    input_data = np.array(keypoints, dtype=np.float32).reshape(1, 21 * 2)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data), np.max(output_data)


# Función para iniciar la captura de video con reconocimiento de señas
def start_static_sign_recognition(label: Label, window, actual_letter):
    global cap, video_update_id
    # Iniciar la captura de la cámara web
    cap = cv2.VideoCapture(id_camera)  # El "id_camera" se cambia em el archivo "config"

    def update_frame():
        global video_update_id
        # Leer un frame de la cámara
        ret, frame = cap.read()
        if ret:
            # Convertir de BGR a RGB (OpenCV usa BGR por defecto)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Si se detecta una mano
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar las conexiones de la mano en el frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    keypoints = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks.landmark]).flatten()
                    # Restar la posición de la muñeca a las coordenadas x e y
                    wrist_x, wrist_y = keypoints[0], keypoints[1]
                    keypoints -= [wrist_x, wrist_y] * 21
                    # Predecir la seña
                    predicted_label_id, confidence = predict_keypoints(keypoints)
                    predicted_label = labels_dict[predicted_label_id]

                    # Actualizar la interfaz según la predicción
                    if confidence > 0.8:  # Si la seña es correcta
                        previous_label = predicted_label
                        update_ui_on_prediction(window, predicted_label, actual_letter)
                        # Cuando se esté en una lección:
                    else:
                        previous_label = None
                        update_ui_on_prediction(window, "low_confidence", actual_letter)
            else:
                # Si no se detecta ninguna mano
                previous_label = None
                update_ui_on_prediction(window, None, actual_letter)

            # Convertir la imagen de OpenCV a un formato que Tkinter puede mostrar
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk  # Necesario para evitar que la imagen sea eliminada por el recolector de basura
            label.config(image=imgtk)

        ## Volver a llamar a esta función después de 10 ms para actualizar el video
        video_update_id = label.after(10, update_frame)

    # Iniciar el bucle de actualización de frames
    update_frame()


# Función para detener la captura de video
def stop_video_stream(label):
    global cap, video_update_id
    if cap:
        cap.release()  # Liberar la cámara
        cap = None

    # Cancelar el ciclo 'after' si está activo
    if video_update_id is not None:
        label.after_cancel(video_update_id)
        video_update_id = None


# Función que actualiza la interfaz dependiendo de la predicción de la seña
def update_ui_on_prediction(window, predicted_label, actual_letter):
    canvas = window.children['!canvas']  # Obtén el canvas de la ventana

    if predicted_label is None:
        # Mostrar un banner si no se detecta ninguna mano
        image_no_hand = PhotoImage(file=relative_to_assets_camera("banner_no_mano.png"))
        canvas.create_image(925.0, 668.5948486328125, image=image_no_hand)
        window.image_no_hand = image_no_hand  # Mantener la referencia de la imagen para evitar que se elimine
    elif predicted_label == actual_letter:
        # Mostrar un banner si la seña es correcta
        image_correct = PhotoImage(file=relative_to_assets_camera("banner_bien.png"))
        canvas.create_image(925.0, 668.5948486328125, image=image_correct)
        window.image_correct = image_correct
    elif predicted_label == "low_confidence":
        # Mostrar un banner si la confianza es baja
        image_low_confidence = PhotoImage(file=relative_to_assets_camera("banner_intento.png"))
        canvas.create_image(925.0, 668.5948486328125, image=image_low_confidence)
        window.image_low_confidence = image_low_confidence
    else:
        # Mostrar un banner si la seña no es correcta
        image_default = PhotoImage(file=relative_to_assets_camera("banner_intento.png"))
        canvas.create_image(925.0, 668.5948486328125, image=image_default)
        window.image_default = image_default
