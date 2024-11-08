# Importación de librerías necesarias para procesamiento de imágenes, manejo de archivos, interfaces, y modelos
import cv2
import json
from PIL import Image, ImageTk
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import pandas as pd
from tkinter import Toplevel, PhotoImage, Label
from Utils.paths import assets_camera_path, generated_models_path, dynamic_model_converted_data_path, \
    static_model_data_labels_path
from Utils.config import static_model_name, dynamic_model_name, id_camera
from GUI.Camera.camera_logic import update_icon_letter

# Definición del alfabeto que el sistema reconocerá
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'NN', 'O',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z']

# Cargar etiquetas para el modelo estático desde un archivo CSV
labels_df = pd.read_csv(static_model_data_labels_path, header=None, index_col=0)
labels_dict = labels_df[1].to_dict()

# Cargar e inicializar el modelo estático en formato TensorFlow Lite
interpreter_static = tf.lite.Interpreter(model_path=os.path.join(generated_models_path, static_model_name))
interpreter_static.allocate_tensors()
input_details_static = interpreter_static.get_input_details()
output_details_static = interpreter_static.get_output_details()

# Cargar el modelo dinámico y obtener los nombres de las acciones disponibles
dynamic_model = load_model(os.path.join(generated_models_path, dynamic_model_name))
actions = [os.path.splitext(action)[0] for action in os.listdir(dynamic_model_converted_data_path) if
           action.endswith(".h5")]

# Configuración de MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Utilidad para dibujar conexiones de la mano


# Función para generar rutas de acceso a los archivos de recursos visuales
def relative_to_assets_camera(path: str) -> Path:
    return assets_camera_path / Path(path)


# Función para cargar la información de la letra actual desde un archivo JSON
def load_letter_data(actual_letter):
    # Cargar datos desde un archivo JSON específico
    DATA_LETTERS_JSON_PATH = "GUI/Assets/Letters_Information/letters_data.json"
    with open(DATA_LETTERS_JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data.get(actual_letter, None)


# Función para realizar predicciones de señas estáticas usando el modelo cargado
def predict_static_sign(keypoints):
    # Preparar los datos de entrada para el modelo estático
    input_data = np.array(keypoints, dtype=np.float32).reshape(1, 21 * 2)
    interpreter_static.set_tensor(input_details_static[0]['index'], input_data)
    interpreter_static.invoke()
    output_data = interpreter_static.get_tensor(output_details_static[0]['index'])
    predicted_index = np.argmax(output_data)
    confidence = np.max(output_data)
    predicted_letter = labels_dict.get(predicted_index, None)
    return predicted_letter, confidence


# Función para mostrar un mensaje de éxito al completar la lección de señas
def show_completion_popup(window):
    popup = Toplevel(window)
    popup.title("¡Completado!")
    window_width, window_height = 665, 665
    screen_width, screen_height = popup.winfo_screenwidth(), popup.winfo_screenheight()
    x_position, y_position = (screen_width // 2) - (window_width // 2), (screen_height // 2) - (window_height // 2)
    popup.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    popup.resizable(False, False)
    completion_image = PhotoImage(file=relative_to_assets_camera("mensaje_exito.png"))
    label = Label(popup, image=completion_image)
    label.pack(padx=10, pady=10)
    popup.completion_image = completion_image
    popup.after(3000, popup.destroy)
    popup.transient(window)
    popup.grab_set()


# Función para iniciar el ciclo de reconocimiento de señas
def start_recognition_cycle(window, video_label, current_letter_index):
    from GUI.Camera.camera_artificial_vision_logic import update_ui_on_prediction
    cap = cv2.VideoCapture(id_camera)
    keypoint_sequence, max_length_frames = [], 15
    last_correct_prediction, current_model_type = None, None
    transitioning = False

    # Función interna que se ejecuta en cada frame capturado
    def update_frame():
        nonlocal current_letter_index, keypoint_sequence, last_correct_prediction, current_model_type, transitioning
        actual_letter = ALPHABET[current_letter_index]
        letter_data = load_letter_data(actual_letter)
        movement_type = letter_data["movement"]

        # Cambiar de modelo (estático o dinámico) dependiendo del tipo de movimiento de la letra
        if movement_type != current_model_type:
            keypoint_sequence.clear()
            last_correct_prediction, current_model_type = None, movement_type

        ret, frame = cap.read()
        if not ret:
            return
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predicción usando el modelo estático
        if movement_type == "Estatico":
            with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
                results = hands.process(image_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        keypoints = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
                        wrist_x, wrist_y = keypoints[0], keypoints[1]
                        keypoints -= [wrist_x, wrist_y] * 21
                        predicted_letter, confidence = predict_static_sign(keypoints)

                        # Verificación de confianza y transición entre letras
                        if confidence > 0.9 and predicted_letter == actual_letter and not transitioning:
                            transitioning = True
                            update_ui_on_prediction(window, predicted_letter, actual_letter)
                            window.after(1000, lambda: transition_to_next_letter())

                        else:
                            last_correct_prediction = None
                        update_ui_on_prediction(window, predicted_letter if confidence > 0.9 else "low_confidence",
                                                actual_letter)
                else:
                    update_ui_on_prediction(window, None, actual_letter)

        # Predicción usando el modelo dinámico
        elif movement_type == "Dinamico":
            with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
                results = hands.process(image_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                        keypoint_sequence.append(keypoints)

                        if len(keypoint_sequence) == max_length_frames:
                            res = dynamic_model.predict(np.expand_dims(keypoint_sequence, axis=0))[0]
                            predicted_action_index = np.argmax(res)
                            predicted_action = actions[predicted_action_index].upper()

                            if res[
                                predicted_action_index] > 0.7 and predicted_action == actual_letter and not transitioning:
                                transitioning = True
                                update_ui_on_prediction(window, predicted_action, actual_letter)
                                window.after(1000, lambda: transition_to_next_letter())

                            else:
                                last_correct_prediction = None
                            keypoint_sequence.clear()
                            update_ui_on_prediction(
                                window,
                                predicted_action if res[predicted_action_index] > 0.7 else "low_confidence",
                                actual_letter)
                else:
                    update_ui_on_prediction(window, None, actual_letter)

        # Actualizar la interfaz gráfica con la imagen del frame procesado
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)
        video_label.after(10, update_frame)

    # Función para avanzar a la siguiente letra
    def transition_to_next_letter():
        nonlocal current_letter_index, last_correct_prediction, transitioning
        current_letter_index = next_letter(window, current_letter_index)
        last_correct_prediction = ALPHABET[current_letter_index]
        transitioning = False

    update_frame()


# Función para avanzar a la siguiente letra y mostrar ventana de éxito si se completa el alfabeto
def next_letter(window, current_letter_index):
    current_letter_index = (current_letter_index + 1) % len(ALPHABET)
    if current_letter_index == 0:
        show_completion_popup(window)

    actual_letter = ALPHABET[current_letter_index]
    letter_data = load_letter_data(actual_letter)
    if letter_data:
        update_icon_letter(window, letter_data["icon_path"])
    return current_letter_index
