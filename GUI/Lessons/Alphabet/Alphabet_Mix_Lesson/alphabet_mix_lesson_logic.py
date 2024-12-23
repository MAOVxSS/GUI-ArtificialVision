# Importación de librerías necesarias para el procesamiento de imágenes, manipulación de archivos,
# interfaces gráficas y modelos de aprendizaje profundo
import os
import cv2
import json
import random
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from mediapipe.python.solutions.holistic import Holistic
import mediapipe as mp
import pandas as pd
from tkinter import Toplevel, PhotoImage, Label
from Utils.paths import assets_camera_path, generated_models_path, static_model_data_labels_path, \
    phrases_model_json_data_path, dynamic_model_keras_path
from Utils.config import static_model_lite_name, dynamic_model_lite_name, id_camera, phrases_to_text, ALPHABET
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

# Inicializar el modelo estático en formato TensorFlow Lite para predicciones rápidas
interpreter_static = tf.lite.Interpreter(model_path=os.path.join(generated_models_path, static_model_lite_name))
interpreter_static.allocate_tensors()
input_details_static = interpreter_static.get_input_details()
output_details_static = interpreter_static.get_output_details()

# Configurar MediaPipe para la detección de manos y dibujo
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Cargar el modelo dinámico en formato TFLite (ya no Keras)
phrases_model_path = os.path.join(dynamic_model_keras_path, dynamic_model_lite_name)
interpreter_dynamic = tf.lite.Interpreter(model_path=phrases_model_path)
interpreter_dynamic.allocate_tensors()
input_details_dynamic = interpreter_dynamic.get_input_details()
output_details_dynamic = interpreter_dynamic.get_output_details()

# Cargar identificadores de palabras desde el archivo JSON
with open(phrases_model_json_data_path, 'r') as json_file:
    data = json.load(json_file)
    word_ids = data.get('word_ids')
    if not word_ids:
        raise ValueError("[ERROR] No se encontraron identificadores de palabras en el archivo JSON.")

# Cargar el scaler
scaler_path = os.path.join(dynamic_model_keras_path, 'scaler.save')
scaler = joblib.load(scaler_path)

def load_letter_data(actual_letter):
    """Carga la información de una letra específica desde un archivo JSON."""
    DATA_LETTERS_JSON_PATH = "GUI/Assets/Letters_Information/letters_data.json"
    with open(DATA_LETTERS_JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data.get(actual_letter, None)

def predict_static_sign(keypoints):
    """Realiza predicciones de señas estáticas utilizando el modelo estático TFLite."""
    input_data = np.array(keypoints, dtype=np.float32).reshape(1, 21 * 2)
    interpreter_static.set_tensor(input_details_static[0]['index'], input_data)
    interpreter_static.invoke()
    output_data = interpreter_static.get_tensor(output_details_static[0]['index'])
    predicted_index = np.argmax(output_data)
    confidence = np.max(output_data)
    predicted_letter = labels_dict.get(predicted_index, None)
    return predicted_letter, confidence

def interpolate_keypoints(keypoints, target_length=15):
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
    cap = cv2.VideoCapture(id_camera)
    keypoint_sequence = []
    last_correct_prediction = None
    transitioning = False
    correct_count = 0
    fix_frames = 0
    count_frame = 0
    recording = False

    holistic_model = Holistic()

    def update_frame():
        nonlocal current_letter_index, keypoint_sequence, last_correct_prediction, transitioning, correct_count
        nonlocal fix_frames, count_frame, recording

        if current_letter_index >= len(MIXED_LETTERS):
            show_completion_popup(window, progress_label, lambda: restart_lesson(window, video_label, progress_label))
            return

        actual_letter = MIXED_LETTERS[current_letter_index]
        letter_data = load_letter_data(actual_letter)
        movement_type = letter_data["movement"]

        ret, frame = cap.read()
        if not ret:
            return
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic_model.process(image_rgb)

        hand_detected = results.left_hand_landmarks or results.right_hand_landmarks
        update_no_hand_banner(window, hand_detected)

        if not hand_detected:
            recording = False
            count_frame = 0
            fix_frames = 0
            keypoint_sequence = []
        else:
            recording = True
            count_frame += 1
            if count_frame > 1:
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

            if movement_type == "Estatico":
                with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
                    results_hands = hands.process(image_rgb)
                    if results_hands.multi_hand_landmarks:
                        for hand_landmarks in results_hands.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            keypoints = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()

                            wrist_x, wrist_y = keypoints[0], keypoints[1]
                            keypoints -= [wrist_x, wrist_y] * 21
                            predicted_letter, confidence = predict_static_sign(keypoints)

                            if confidence > 0.9 and predicted_letter == actual_letter and not transitioning:
                                transitioning = True
                                update_ui_on_prediction(window, predicted_letter, actual_letter)
                                correct_count += 1
                                progress_label.config(text=f"{correct_count}/10")
                                window.after(2000, lambda: transition_to_next_letter())
                            else:
                                last_correct_prediction = None
                            update_ui_on_prediction(window, predicted_letter if confidence > 0.9 else "low_confidence",
                                                    actual_letter)
                    else:
                        update_ui_on_prediction(window, None, actual_letter)

            elif movement_type == "Dinamico":
                if count_frame >= MIN_LENGTH_FRAMES + 1:
                    fix_frames += 1
                    if fix_frames >= 3:
                        recording = False
                        keypoint_sequence = keypoint_sequence[: - (1 + 3)]
                        kp_normalized = normalize_keypoints(keypoint_sequence, int(MODEL_FRAMES))
                        kp_normalized = np.array(kp_normalized)
                        num_frames, num_keypoints = kp_normalized.shape

                        kp_flat = kp_normalized.reshape(-1, num_keypoints)
                        kp_scaled = scaler.transform(kp_flat)
                        kp_scaled = kp_scaled.reshape(num_frames, num_keypoints).astype(np.float32)

                        # Predecir con el modelo dinámico TFLite
                        interpreter_dynamic.resize_tensor_input(input_details_dynamic[0]['index'],
                                                                [1, MODEL_FRAMES, num_keypoints])
                        interpreter_dynamic.allocate_tensors()
                        interpreter_dynamic.set_tensor(input_details_dynamic[0]['index'], kp_scaled[np.newaxis, ...])
                        interpreter_dynamic.invoke()
                        res = interpreter_dynamic.get_tensor(output_details_dynamic[0]['index'])[0]

                        pred_index = np.argmax(res)
                        confidence = res[pred_index] * 100

                        if confidence > 50:
                            predicted_word = word_ids[pred_index]
                            general_word = phrases_to_text.get(predicted_word, predicted_word)
                            actual_letter_lower = actual_letter.lower()

                            print(f"\nPredicción detectada:")
                            print(f"- Palabra predicha (original): {predicted_word}")
                            print(f"- Palabra predicha (general): {general_word}")
                            print(f"- Confianza: {confidence:.2f}%")
                            print(f"- Índice del modelo: {pred_index}")

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

                        count_frame = 0
                        fix_frames = 0
                        keypoint_sequence = []
                    else:
                        recording = True

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)
        video_label.after(10, update_frame)

    def transition_to_next_letter():
        nonlocal current_letter_index, last_correct_prediction, transitioning
        current_letter_index = next_letter(window, video_label, current_letter_index, progress_label)
        if current_letter_index < len(MIXED_LETTERS):
            last_correct_prediction = MIXED_LETTERS[current_letter_index]
        transitioning = False

    update_frame()

def next_letter(window, video_label, current_letter_index, progress_label):
    current_letter_index += 1
    if current_letter_index >= len(MIXED_LETTERS):
        return current_letter_index

    actual_letter = MIXED_LETTERS[current_letter_index]
    letter_data = load_letter_data(actual_letter)
    if letter_data:
        update_icon_letter(window, letter_data["icon_path"])
    return current_letter_index

def restart_lesson(window, video_label, progress_label):
    global MIXED_LETTERS
    MIXED_LETTERS = random.sample(ALPHABET, 10)
    if isinstance(progress_label, Label):
        progress_label.config(text="0/10")
    start_letters_mix_recognition_cycle(window, video_label, progress_label)
