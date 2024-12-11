# Importación de librerías necesarias para el procesamiento de imágenes, manipulación de archivos,
# interfaces gráficas y modelos de aprendizaje profundo
import os
import cv2
import json
import random
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from mediapipe.python.solutions.holistic import Holistic
import mediapipe as mp
import pandas as pd
from tkinter import Toplevel, PhotoImage, Label
from Utils.paths import assets_camera_path, generated_models_path, static_model_data_labels_path, \
    phrases_model_json_data_path, phrases_model_keras_path
from Utils.config import static_model_name, phrases_model_keras_name, id_camera, phrases_to_text, ALPHABET
from GUI.Camera.Camera_Letters.camera_letters_logic import update_icon_letter
from GUI.Camera.Camera_Letters.camera_letters_model_logic import relative_to_assets_camera, update_ui_on_prediction
from GUI.Camera.Camera_Phrases.camera_phrases_model_logic import update_no_hand_banner, update_ui_on_prediction_phrases
from GUI.gui_utils import show_completion_popup
import joblib

# Constants
MODEL_FRAMES = 15
MIN_LENGTH_FRAMES = 5

# Seleccionar 10 letras aleatorias del alfabeto
MIXED_LETTERS = random.sample(ALPHABET, 10)

# Cargar etiquetas para el modelo estático desde el archivo CSV especificado
labels_df = pd.read_csv(static_model_data_labels_path, header=None, index_col=0)
labels_dict = labels_df[1].to_dict()

# Inicializar el modelo estático en formato TensorFlow Lite para predicciones rápidas en dispositivos ligeros
interpreter_static = tf.lite.Interpreter(model_path=os.path.join(generated_models_path, static_model_name))
interpreter_static.allocate_tensors()
input_details_static = interpreter_static.get_input_details()
output_details_static = interpreter_static.get_output_details()

# Configurar MediaPipe para la detección de manos y la creación de puntos de referencia
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Herramienta para dibujar conexiones entre puntos de referencia de la mano

# Cargar el modelo dinámico y los datos necesarios
phrases_model_path = os.path.join(phrases_model_keras_path, phrases_model_keras_name)
dynamic_model = load_model(phrases_model_path)

# Cargar identificadores de palabras desde el archivo JSON
with open(phrases_model_json_data_path, 'r') as json_file:
    data = json.load(json_file)
    word_ids = data.get('word_ids')
    if not word_ids:
        raise ValueError("[ERROR] No se encontraron identificadores de palabras en el archivo JSON.")

# Cargar el scaler
scaler_path = os.path.join(phrases_model_keras_path, 'scaler.save')
scaler = joblib.load(scaler_path)

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

def interpolate_keypoints(keypoints, target_length=15):
    """
    Interpola los puntos clave para ajustarlos al tamaño deseado.
    """
    current_length = len(keypoints)
    if current_length == target_length:
        return keypoints

    indices = np.linspace(0, current_length - 1, target_length)
    interpolated_keypoints = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        if lower_idx == upper_idx:
            interpolated_keypoints.append(keypoints[lower_idx])
        else:
            interpolated_point = (1 - weight) * np.array(keypoints[lower_idx]) + weight * np.array(keypoints[upper_idx])
            interpolated_keypoints.append(interpolated_point.tolist())

    return interpolated_keypoints

def normalize_keypoints(keypoints, target_length=15):
    """
    Normaliza la longitud de los puntos clave para ajustarla al tamaño objetivo.
    """
    current_length = len(keypoints)
    if current_length < target_length:
        return interpolate_keypoints(keypoints, target_length)
    elif current_length > target_length:
        step = current_length / target_length
        indices = np.arange(0, current_length, step).astype(int)[:target_length]
        return [keypoints[i] for i in indices]
    else:
        return keypoints

def start_letters_mix_recognition_cycle(window, video_label, progress_label, current_letter_index=0):
    """Inicia el ciclo de reconocimiento de señas y actualiza la interfaz con los resultados.

    Argumentos:
    window -- ventana principal de la aplicación.
    video_label -- etiqueta donde se mostrará el video en tiempo real.
    progress_label -- etiqueta que muestra el progreso de la lección.
    current_letter_index -- índice de la letra actual en el alfabeto mezclado (por defecto 0).
    """
    cap = cv2.VideoCapture(id_camera)  # Captura de video desde la cámara
    keypoint_sequence = []  # Inicializar secuencia de puntos clave
    last_correct_prediction = None
    transitioning = False  # Estado de transición entre letras
    correct_count = 0  # Contador de letras correctas para el progreso
    fix_frames = 0
    count_frame = 0
    recording = False

    # Inicializar MediaPipe Holistic para la detección de puntos clave
    holistic_model = Holistic()

    def update_frame():
        """Captura y procesa cada frame del video para identificar señas."""
        nonlocal current_letter_index, keypoint_sequence, last_correct_prediction, transitioning, correct_count
        nonlocal fix_frames, count_frame, recording

        # Verificar si se han completado todas las letras
        if current_letter_index >= len(MIXED_LETTERS):
            show_completion_popup(window, progress_label, lambda: restart_lesson(window, video_label, progress_label))
            return

        # Obtener la letra actual y sus datos correspondientes
        actual_letter = MIXED_LETTERS[current_letter_index]
        letter_data = load_letter_data(actual_letter)
        movement_type = letter_data["movement"]

        # Leer el frame de la cámara; si falla, finalizar la actualización
        ret, frame = cap.read()
        if not ret:
            return
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir el frame a RGB para el procesamiento

        # Procesar el frame con MediaPipe
        image_rgb.flags.writeable = False
        results = holistic_model.process(image_rgb)

        # Detectar manos
        hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

        # Actualizar el banner de 'No se detecta ninguna mano'
        update_no_hand_banner(window, hand_detected)

        if not hand_detected:
            # Reiniciar variables si es necesario
            recording = False
            count_frame = 0
            fix_frames = 0
            keypoint_sequence = []
        else:
            # Captura los frames mientras haya manos detectadas
            recording = True
            count_frame += 1
            if count_frame > 1:  # margin_frame = 1
                pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                                 results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(
                    33 * 4)
                face = np.array([[res.x, res.y, res.z] for res in
                                 results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(
                    468 * 3)
                left_hand = np.array([[res.x, res.y, res.z] for res in
                                      results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
                    21 * 3)
                right_hand = np.array([[res.x, res.y, res.z] for res in
                                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
                    21 * 3)
                kp_frame = np.concatenate([pose, face, left_hand, right_hand])
                keypoint_sequence.append(kp_frame)

            # Si el movimiento es estático, realizar la predicción con el modelo estático
            if movement_type == "Estatico":
                # Procesar con MediaPipe Hands
                with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
                    results_hands = hands.process(image_rgb)
                    if results_hands.multi_hand_landmarks:
                        for hand_landmarks in results_hands.multi_hand_landmarks:
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
                                progress_label.config(text=f"{correct_count}/10")
                                window.after(2000, lambda: transition_to_next_letter())
                            else:
                                last_correct_prediction = None
                            update_ui_on_prediction(window, predicted_letter if confidence > 0.9 else "low_confidence",
                                                    actual_letter)
                    else:
                        update_ui_on_prediction(window, None, actual_letter)  # Si no hay detección, actualizar con None

            # Si el movimiento es dinámico, usar el modelo dinámico
            elif movement_type == "Dinamico":
                # Verificar si se ha completado la secuencia para predecir
                if count_frame >= MIN_LENGTH_FRAMES + 1:  # margin_frame = 1
                    fix_frames += 1
                    if fix_frames >= 3:  # delay_frames = 3
                        recording = False
                        keypoint_sequence = keypoint_sequence[: - (1 + 3)]  # margin_frame + delay_frames
                        kp_normalized = normalize_keypoints(keypoint_sequence, int(MODEL_FRAMES))
                        kp_normalized = np.array(kp_normalized)
                        num_frames, num_keypoints = kp_normalized.shape

                        # Aplicar el scaler
                        kp_flat = kp_normalized.reshape(-1, num_keypoints)
                        kp_scaled = scaler.transform(kp_flat)
                        kp_scaled = kp_scaled.reshape(num_frames, num_keypoints)

                        res = dynamic_model.predict(np.expand_dims(kp_scaled, axis=0))[0]

                        # Obtener el índice y valor de la predicción máxima
                        pred_index = np.argmax(res)
                        confidence = res[pred_index] * 100

                        if confidence > 50:  # threshold = 0.5
                            # Obtener la predicción original
                            predicted_word = word_ids[pred_index]

                            # Mapear la predicción a su forma general utilizando phrases_to_text
                            general_word = phrases_to_text.get(predicted_word, predicted_word)

                            # Convertir actual_letter a minúsculas para comparación
                            actual_letter_lower = actual_letter.lower()

                            print(f"\nPredicción detectada:")
                            print(f"- Palabra predicha (original): {predicted_word}")
                            print(f"- Palabra predicha (general): {general_word}")
                            print(f"- Confianza: {confidence:.2f}%")
                            print(f"- Índice del modelo: {pred_index}")

                            # Actualizar UI basado en la predicción general
                            if general_word == actual_letter_lower and not transitioning:
                                print("¡CORRECTO! La seña coincide con el objetivo")
                                transitioning = True
                                update_ui_on_prediction_phrases(window, actual_letter, actual_letter)
                                correct_count += 1
                                progress_label.config(text=f"{correct_count}/10")
                                window.after(2000, lambda: transition_to_next_letter())
                            else:
                                print(f"INCORRECTO: La seña no coincide con el objetivo")
                                print(f"- Esperado: {actual_letter_lower}")
                                print(f"- Recibido: {general_word}")
                                update_ui_on_prediction_phrases(window, "incorrect", actual_letter)
                        else:
                            print(f"\nBaja confianza en la predicción:")
                            print(f"- Confianza: {confidence:.2f}%")
                            print(f"- Umbral requerido: 50%")
                            update_ui_on_prediction_phrases(window, "low_confidence", actual_letter)

                        # Resetea los estados
                        count_frame = 0
                        fix_frames = 0
                        keypoint_sequence = []
                    else:
                        recording = True  # Continuar grabando hasta alcanzar delay_frames

        # Convertir el frame procesado a imagen y actualizar en la interfaz gráfica
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk  # Mantener la referencia de la imagen para evitar recolección de basura
        video_label.config(image=imgtk)
        video_label.after(10, update_frame)  # Repetir la actualización del frame

    def transition_to_next_letter():
        """Avanza a la siguiente letra en la lección y reinicia la transición."""
        nonlocal current_letter_index, last_correct_prediction, transitioning
        current_letter_index = next_letter(window, video_label, current_letter_index, progress_label)
        if current_letter_index < len(MIXED_LETTERS):
            last_correct_prediction = MIXED_LETTERS[current_letter_index]
        transitioning = False  # Reiniciar el estado de transición

    update_frame()  # Iniciar el ciclo de actualización de frames

def next_letter(window, video_label, current_letter_index, progress_label):
    """Avanza el índice de la letra actual y actualiza la interfaz."""
    current_letter_index += 1
    if current_letter_index >= len(MIXED_LETTERS):
        # No hacemos nada aquí porque la comprobación se hace en update_frame
        return current_letter_index

    # Cargar los datos de la letra siguiente y actualizar el icono en la UI
    actual_letter = MIXED_LETTERS[current_letter_index]
    letter_data = load_letter_data(actual_letter)
    if letter_data:
        update_icon_letter(window, letter_data["icon_path"])
    return current_letter_index

def restart_lesson(window, video_label, progress_label):
    """Reinicia la lección al azar seleccionando nuevas letras y actualizando la interfaz.

    Argumentos:
    window -- Ventana principal de la aplicación.
    video_label -- Etiqueta donde se muestra el video en tiempo real.
    progress_label -- Etiqueta que muestra el progreso de la lección.
    """
    global MIXED_LETTERS
    MIXED_LETTERS = random.sample(ALPHABET, 10)  # Seleccionar nuevas letras al azar
    if isinstance(progress_label, Label):
        progress_label.config(text="0/10")
    start_letters_mix_recognition_cycle(window, video_label, progress_label)
