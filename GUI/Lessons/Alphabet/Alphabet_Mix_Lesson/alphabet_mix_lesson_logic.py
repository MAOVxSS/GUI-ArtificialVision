# Importación de librerías necesarias para el procesamiento de imágenes, manipulación de archivos,
# interfaces gráficas y modelos de aprendizaje profundo
import cv2
import json
import random
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

# Cargar el alfabeto completo y seleccionar aleatoriamente 10 letras para la lección
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'NN', 'O',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
            'Y', 'Z']
MIXED_LETTERS = random.sample(ALPHABET, 10)  # Selección inicial de 10 letras al azar

# Cargar etiquetas para el modelo estático desde un archivo CSV
labels_df = pd.read_csv(static_model_data_labels_path, header=None, index_col=0)
labels_dict = labels_df[1].to_dict()

# Cargar el modelo estático e inicializar los tensores de entrada y salida
interpreter_static = tf.lite.Interpreter(model_path=os.path.join(generated_models_path, static_model_name))
interpreter_static.allocate_tensors()
input_details_static = interpreter_static.get_input_details()
output_details_static = interpreter_static.get_output_details()

# Cargar el modelo dinámico y obtener nombres de acciones disponibles
dynamic_model = load_model(os.path.join(generated_models_path, dynamic_model_name))
actions = [os.path.splitext(action)[0] for action in os.listdir(dynamic_model_converted_data_path) if
           action.endswith(".h5")]

# Configuración de MediaPipe para la detección de manos y dibujo de puntos clave en la interfaz
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Herramienta para dibujar conexiones de la mano


# Generar rutas completas a los archivos de recursos visuales
def relative_to_assets_camera(path: str) -> Path:
    """Devuelve la ruta completa hacia un recurso gráfico dado un path relativo."""
    return assets_camera_path / Path(path)


# Cargar datos de una letra específica desde un archivo JSON
def load_letter_data(actual_letter):
    """Carga datos de una letra específica desde un archivo JSON.

    Argumentos:
    actual_letter -- La letra actual para la cual se cargan los datos.

    Retorna:
    Diccionario con información de la letra si está disponible.
    """
    DATA_LETTERS_JSON_PATH = "GUI/Assets/Letters_Information/letters_data.json"
    with open(DATA_LETTERS_JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data.get(actual_letter, None)


# Realiza predicciones de señas estáticas utilizando el modelo cargado
def predict_static_sign(keypoints):
    """Predice la letra de señas en el modelo estático usando puntos clave.

    Argumentos:
    keypoints -- Coordenadas de los puntos clave de la mano.

    Retorna:
    predicted_letter -- Letra predicha por el modelo.
    confidence -- Nivel de confianza de la predicción.
    """
    input_data = np.array(keypoints, dtype=np.float32).reshape(1, 21 * 2)  # Preparar los puntos clave como entrada
    interpreter_static.set_tensor(input_details_static[0]['index'], input_data)  # Asignar datos de entrada
    interpreter_static.invoke()  # Ejecutar el modelo
    output_data = interpreter_static.get_tensor(output_details_static[0]['index'])  # Obtener predicción
    predicted_index = np.argmax(output_data)  # Índice de la letra con mayor probabilidad
    confidence = np.max(output_data)  # Nivel de confianza de la predicción
    predicted_letter = labels_dict.get(predicted_index, None)  # Obtener la letra correspondiente
    return predicted_letter, confidence


# Mostrar ventana emergente de éxito al completar la lección y reiniciar la lección
def show_completion_popup(window, progress_label, restart_lesson_callback):
    """Muestra una ventana emergente de éxito y reinicia la lección.

    Argumentos:
    window -- Ventana principal de la aplicación.
    progress_label -- Etiqueta de progreso de la lección.
    restart_lesson_callback -- Función a ejecutar para reiniciar la lección.
    """
    # Crear la ventana emergente centrada en la pantalla
    popup = Toplevel(window)
    popup.title("¡Completado!")
    window_width, window_height = 665, 665
    screen_width, screen_height = popup.winfo_screenwidth(), popup.winfo_screenheight()
    x_position, y_position = (screen_width // 2) - (window_width // 2), (screen_height // 2) - (window_height // 2)
    popup.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    popup.resizable(False, False)

    # Cargar imagen de éxito y asignarla a la ventana emergente
    completion_image = PhotoImage(file=relative_to_assets_camera("mensaje_exito.png"))
    label = Label(popup, image=completion_image)
    label.pack(padx=10, pady=10)
    popup.completion_image = completion_image  # Mantener referencia a la imagen para evitar recolección de basura

    # Cerrar la ventana emergente después de 3 segundos y reiniciar la lección
    popup.after(3000, lambda: [popup.destroy(), restart_lesson_callback()])
    popup.transient(window)  # Definir la ventana emergente como hija de la ventana principal
    popup.grab_set()  # Bloquear interacción con la ventana principal mientras la emergente esté activa


# Inicia el ciclo de reconocimiento de señas para las letras seleccionadas al azar
def start_recognition_cycle(window, video_label, progress_label, current_letter_index=0):
    """Inicia el ciclo de reconocimiento para las letras seleccionadas al azar.

    Argumentos:
    window -- Ventana principal de la aplicación.
    video_label -- Etiqueta donde se muestra el video en tiempo real.
    progress_label -- Etiqueta que muestra el progreso de la lección.
    current_letter_index -- Índice de la letra actual en el alfabeto mezclado (por defecto 0).
    """
    from GUI.Camera.camera_artificial_vision_logic import update_ui_on_prediction
    cap = cv2.VideoCapture(id_camera)  # Captura de video desde la cámara
    keypoint_sequence, max_length_frames = [], 15  # Secuencia de puntos clave y número de frames para predicción
    last_correct_prediction, current_model_type = None, None  # Estado inicial del modelo y predicción
    transitioning = False  # Estado de transición entre letras
    correct_count = 0  # Contador de letras correctas para el progreso

    def update_frame():
        """Captura y procesa cada frame del video para identificar señas."""
        nonlocal current_letter_index, keypoint_sequence, last_correct_prediction, current_model_type, transitioning, correct_count

        # Obtener la letra actual del conjunto mezclado y sus datos específicos
        actual_letter = MIXED_LETTERS[current_letter_index]
        letter_data = load_letter_data(actual_letter)
        movement_type = letter_data["movement"]  # Obtener el tipo de movimiento de la letra

        # Cambiar entre modelos si el tipo de movimiento de la letra actual es distinto al del modelo en uso
        if movement_type != current_model_type:
            keypoint_sequence.clear()  # Limpiar la secuencia de puntos clave si cambia el modelo
            last_correct_prediction, current_model_type = None, movement_type

        # Leer el frame de la cámara; si falla, salir de la actualización
        ret, frame = cap.read()
        if not ret:
            return
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir el frame a RGB para procesamiento

        # Predicción con el modelo estático si el movimiento es "Estatico"
        if movement_type == "Estatico":
            with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
                results = hands.process(image_rgb)
                if results.multi_hand_landmarks:  # Si se detectan manos en el frame
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Dibujar puntos
                        keypoints = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()

                        # Normalizar los puntos con respecto a la muñeca
                        wrist_x, wrist_y = keypoints[0], keypoints[1]
                        keypoints -= [wrist_x, wrist_y] * 21
                        predicted_letter, confidence = predict_static_sign(keypoints)

                        # Verificar si la predicción es correcta y confiable
                        if confidence > 0.9 and predicted_letter == actual_letter and not transitioning:
                            transitioning = True  # Indicar que se está en transición hacia la siguiente letra
                            update_ui_on_prediction(window, predicted_letter, actual_letter)
                            correct_count += 1  # Incrementar el contador de letras correctas
                            if isinstance(progress_label, Label):  # Verificar si el progreso es un Label
                                progress_label.config(text=f"{correct_count}/10")  # Actualizar el progreso
                            window.after(1000, lambda: transition_to_next_letter())  # Pasar a la siguiente letra
                        else:
                            last_correct_prediction = None
                        update_ui_on_prediction(window, predicted_letter if confidence > 0.9 else "low_confidence",
                                                actual_letter)
                else:
                    update_ui_on_prediction(window, None, actual_letter)  # Actualizar la UI si no se detectan manos

        # Predicción con el modelo dinámico si el movimiento es "Dinamico"
        elif movement_type == "Dinamico":
            with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
                results = hands.process(image_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                        keypoint_sequence.append(keypoints)

                        # Si se completa la secuencia, realizar predicción con el modelo dinámico
                        if len(keypoint_sequence) == max_length_frames:
                            res = dynamic_model.predict(np.expand_dims(keypoint_sequence, axis=0))[0]
                            predicted_action_index = np.argmax(res)
                            predicted_action = actions[predicted_action_index].upper()

                            # Verificar si la predicción es correcta y confiable
                            if res[
                                predicted_action_index] > 0.7 and predicted_action == actual_letter and not transitioning:
                                transitioning = True
                                update_ui_on_prediction(window, predicted_action, actual_letter)
                                correct_count += 1
                                if isinstance(progress_label, Label):  # Verificar si el progreso es un Label
                                    progress_label.config(text=f"{correct_count}/10")
                                window.after(1000, lambda: transition_to_next_letter())  # Pasar a la siguiente letra
                            else:
                                last_correct_prediction = None
                            keypoint_sequence.clear()  # Limpiar secuencia de puntos clave
                            update_ui_on_prediction(window, predicted_action if res[
                                                                                    predicted_action_index] > 0.7 else "low_confidence",
                                                    actual_letter)
                else:
                    update_ui_on_prediction(window, None, actual_letter)  # Actualizar si no hay detección de mano

        # Convertir el frame procesado a imagen y actualizar en la interfaz gráfica
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk  # Mantener referencia de la imagen para evitar recolección de basura
        video_label.config(image=imgtk)
        video_label.after(10, update_frame)  # Repetir la actualización del frame

    def transition_to_next_letter():
        """Avanza a la siguiente letra en la lección y reinicia la transición."""
        nonlocal current_letter_index, last_correct_prediction, transitioning
        current_letter_index = next_letter(window, current_letter_index, progress_label)
        last_correct_prediction = MIXED_LETTERS[current_letter_index]
        transitioning = False  # Reiniciar estado de transición

    update_frame()  # Iniciar el ciclo de actualización de frames


def next_letter(window, current_letter_index, progress_label):
    """Avanza a la siguiente letra y reinicia al completar las 10 letras seleccionadas."""
    current_letter_index += 1
    if current_letter_index >= len(MIXED_LETTERS):
        show_completion_popup(window, progress_label, lambda: restart_lesson(window, progress_label))
        return 0  # Reiniciar el índice al final

    # Cargar datos de la siguiente letra y actualizar el icono en la UI
    actual_letter = MIXED_LETTERS[current_letter_index]
    letter_data = load_letter_data(actual_letter)
    if letter_data:
        update_icon_letter(window, letter_data["icon_path"])
    return current_letter_index


def restart_lesson(window, progress_label):
    """Reinicia la lección al azar seleccionando nuevas letras y actualizando la interfaz.

    Argumentos:
    window -- Ventana principal de la aplicación.
    progress_label -- Etiqueta que muestra el progreso de la lección.
    """
    global MIXED_LETTERS
    MIXED_LETTERS = random.sample(ALPHABET, 10)  # Seleccionar nuevas letras al azar
    if isinstance(progress_label, Label):  # Verificar si progress_label es un Label
        progress_label.config(text="0/10")  # Reiniciar el contador de progreso
    start_recognition_cycle(window, progress_label)  # Iniciar el nuevo ciclo de reconocimiento
