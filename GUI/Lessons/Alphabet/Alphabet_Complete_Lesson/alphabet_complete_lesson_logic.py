# Importación de librerías necesarias para el procesamiento de imágenes, manipulación de archivos, interfaces gráficas y modelos de aprendizaje profundo.
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

# Definición del alfabeto reconocido por el sistema de señas
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'NN', 'O',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z']

# Cargar etiquetas para el modelo estático desde el archivo CSV especificado
labels_df = pd.read_csv(static_model_data_labels_path, header=None, index_col=0)
labels_dict = labels_df[1].to_dict()

# Inicializar el modelo estático en formato TensorFlow Lite para predicciones rápidas en dispositivos ligeros
interpreter_static = tf.lite.Interpreter(model_path=os.path.join(generated_models_path, static_model_name))
interpreter_static.allocate_tensors()
input_details_static = interpreter_static.get_input_details()
output_details_static = interpreter_static.get_output_details()

# Cargar el modelo dinámico y extraer los nombres de las acciones de los archivos convertidos
dynamic_model = load_model(os.path.join(generated_models_path, dynamic_model_name))
actions = [os.path.splitext(action)[0] for action in os.listdir(dynamic_model_converted_data_path) if
           action.endswith(".h5")]

# Configurar MediaPipe para la detección de manos y la creación de puntos de referencia
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Herramienta para dibujar conexiones entre puntos de referencia de la mano


def relative_to_assets_camera(path: str) -> Path:
    """Devuelve la ruta completa hacia un recurso gráfico dado un path relativo."""
    return assets_camera_path / Path(path)


def load_letter_data(actual_letter):
    """Carga la información de una letra específica desde un archivo JSON.

    Argumentos:
    actual_letter -- la letra actual para la cual se cargan los datos.

    Retorna:
    Diccionario con información de la letra si está disponible.
    """
    DATA_LETTERS_JSON_PATH = "GUI/Assets/Letters_Information/letters_data.json"
    with open(DATA_LETTERS_JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data.get(actual_letter, None)  # Retorna datos de la letra si están disponibles


def predict_static_sign(keypoints):
    """Realiza predicciones de señas estáticas utilizando el modelo estático.

    Argumentos:
    keypoints -- coordenadas de los puntos de referencia de la mano.

    Retorna:
    predicted_letter -- letra predicha por el modelo.
    confidence -- nivel de confianza de la predicción.
    """
    # Preparar la entrada del modelo: convertir los puntos de referencia en un array de floats
    input_data = np.array(keypoints, dtype=np.float32).reshape(1, 21 * 2)

    # Asignar los datos de entrada al tensor del modelo y realizar la predicción
    interpreter_static.set_tensor(input_details_static[0]['index'], input_data)
    interpreter_static.invoke()

    # Obtener el resultado de la predicción y determinar la letra y la confianza
    output_data = interpreter_static.get_tensor(output_details_static[0]['index'])
    predicted_index = np.argmax(output_data)
    confidence = np.max(output_data)
    predicted_letter = labels_dict.get(predicted_index, None)  # Buscar la letra predicha en el diccionario de etiquetas

    return predicted_letter, confidence


def show_completion_popup(window, progress_label, restart_lesson_callback):
    """Muestra una ventana emergente de éxito al completar la lección.

    Argumentos:
    window -- ventana principal de la aplicación.
    progress_label -- etiqueta de progreso de la lección.
    restart_lesson_callback -- función a ejecutar para reiniciar la lección.
    """
    # Crear la ventana emergente centrada en la pantalla
    popup = Toplevel(window)
    popup.title("¡Completado!")
    window_width, window_height = 665, 665
    screen_width, screen_height = popup.winfo_screenwidth(), popup.winfo_screenheight()
    x_position, y_position = (screen_width // 2) - (window_width // 2), (screen_height // 2) - (window_height // 2)
    popup.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    popup.resizable(False, False)

    # Cargar la imagen de éxito y asignarla a la ventana emergente
    completion_image = PhotoImage(file=relative_to_assets_camera("mensaje_exito.png"))
    label = Label(popup, image=completion_image)
    label.pack(padx=10, pady=10)
    popup.completion_image = completion_image  # Mantener la referencia a la imagen para evitar recolección de basura

    # Configurar el cierre automático de la ventana emergente después de 3 segundos y reiniciar la lección
    popup.after(3000, lambda: [popup.destroy(), restart_lesson_callback()])
    popup.transient(window)  # Establecer la ventana emergente como hija de la ventana principal
    popup.grab_set()  # Bloquear interacción con la ventana principal mientras la emergente esté activa


def start_recognition_cycle(window, video_label, progress_label, current_letter_index=0):
    """Inicia el ciclo de reconocimiento de señas y actualiza la interfaz con los resultados.

    Argumentos:
    window -- ventana principal de la aplicación.
    video_label -- etiqueta donde se mostrará el video en tiempo real.
    progress_label -- etiqueta que muestra el progreso de la lección.
    current_letter_index -- índice de la letra actual en el alfabeto (por defecto 0).
    """
    from GUI.Camera.camera_artificial_vision_logic import update_ui_on_prediction
    cap = cv2.VideoCapture(id_camera)  # Captura de video desde la cámara
    keypoint_sequence, max_length_frames = [], 15  # Inicializar secuencia de puntos clave y longitud máxima de frames
    last_correct_prediction, current_model_type = None, None
    transitioning = False  # Estado de transición entre letras
    correct_count = 0  # Contador de letras correctas para el progreso

    def update_frame():
        """Captura y procesa cada frame del video para identificar señas."""
        nonlocal current_letter_index, keypoint_sequence, last_correct_prediction, current_model_type, transitioning, correct_count

        # Obtener la letra actual del alfabeto y sus datos correspondientes
        actual_letter = ALPHABET[current_letter_index]
        letter_data = load_letter_data(actual_letter)
        movement_type = letter_data["movement"]

        # Cambiar entre modelos estático y dinámico según el tipo de movimiento requerido para la letra
        if movement_type != current_model_type:
            keypoint_sequence.clear()  # Limpiar la secuencia de puntos clave si cambia el tipo de modelo
            last_correct_prediction, current_model_type = None, movement_type

        # Leer el frame de la cámara; si falla, finalizar la actualización
        ret, frame = cap.read()
        if not ret:
            return
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir el frame a RGB para el procesamiento

        # Si el movimiento es estático, realizar la predicción con el modelo estático
        if movement_type == "Estatico":
            with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
                results = hands.process(image_rgb)  # Procesar la imagen para detectar manos
                if results.multi_hand_landmarks:  # Si se detectan manos en la imagen
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Dibujar puntos
                        keypoints = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()

                        # Normalizar los puntos respecto a la muñeca
                        wrist_x, wrist_y = keypoints[0], keypoints[1]
                        keypoints -= [wrist_x, wrist_y] * 21
                        predicted_letter, confidence = predict_static_sign(keypoints)

                        # Actualizar la UI si la predicción es correcta y confiable
                        if confidence > 0.9 and predicted_letter == actual_letter and not transitioning:
                            transitioning = True
                            update_ui_on_prediction(window, predicted_letter, actual_letter)
                            correct_count += 1  # Aumentar el contador de progreso
                            progress_label.config(text=f"{correct_count}/27")
                            window.after(1000, lambda: transition_to_next_letter())  # Mover a la siguiente letra
                        else:
                            last_correct_prediction = None
                        update_ui_on_prediction(window, predicted_letter if confidence > 0.9 else "low_confidence",
                                                actual_letter)
                else:
                    update_ui_on_prediction(window, None, actual_letter)  # Si no hay detección, actualizar con None

        # Si el movimiento es dinámico, usar el modelo dinámico
        elif movement_type == "Dinamico":
            with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
                results = hands.process(image_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                        keypoint_sequence.append(keypoints)  # Agregar puntos clave a la secuencia

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
                                progress_label.config(text=f"{correct_count}/27")
                                window.after(1000, lambda: transition_to_next_letter())
                            else:
                                last_correct_prediction = None
                            keypoint_sequence.clear()  # Limpiar la secuencia para el próximo ciclo
                            update_ui_on_prediction(
                                window,
                                predicted_action if res[predicted_action_index] > 0.7 else "low_confidence",
                                actual_letter)
                else:
                    update_ui_on_prediction(window, None, actual_letter)  # Actualizar si no hay detección de mano

        # Convertir el frame procesado a imagen y actualizar en la interfaz gráfica
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk  # Mantener la referencia de la imagen para evitar recolección de basura
        video_label.config(image=imgtk)
        video_label.after(10, update_frame)  # Repetir la actualización del frame

    def transition_to_next_letter():
        """Avanza a la siguiente letra en la lección y reinicia la transición."""
        nonlocal current_letter_index, last_correct_prediction, transitioning
        current_letter_index = next_letter(window, current_letter_index, progress_label)
        last_correct_prediction = ALPHABET[current_letter_index]
        transitioning = False  # Reiniciar el estado de transición

    update_frame()  # Iniciar el ciclo de actualización de frames


def next_letter(window, current_letter_index, progress_label):
    """Avanza el índice de la letra actual y muestra una ventana de éxito al completar el abecedario."""
    current_letter_index += 1
    if current_letter_index >= len(ALPHABET):
        show_completion_popup(window, progress_label, lambda: restart_lesson(window, progress_label))
        return 0

    # Cargar los datos de la letra siguiente y actualizar el icono en la UI
    actual_letter = ALPHABET[current_letter_index]
    letter_data = load_letter_data(actual_letter)
    if letter_data:
        update_icon_letter(window, letter_data["icon_path"])
    return current_letter_index


def restart_lesson(window, progress_label):
    """Reinicia la lección, restableciendo el progreso y comenzando el ciclo de reconocimiento desde la letra A."""
    if isinstance(progress_label, Label):
        progress_label.config(text="0/27")
    start_recognition_cycle(window, progress_label)
