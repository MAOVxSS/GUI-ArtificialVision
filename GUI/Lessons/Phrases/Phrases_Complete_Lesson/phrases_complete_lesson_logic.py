# Importación de librerías necesarias para el procesamiento de imágenes, manipulación de archivos, interfaces gráficas y modelos de aprendizaje profundo.
import os
import cv2
import json
import pandas as pd
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from mediapipe.python.solutions.holistic import Holistic
from tkinter import Label
from Utils.paths import phrases_model_json_data_path, phrases_model_keras_path
from Utils.config import phrases_model_lite_name, id_camera, phrases_to_text, PHRASES
from GUI.Camera.Camera_Letters.camera_letters_logic import update_icon_letter
from GUI.Camera.Camera_Phrases.camera_phrases_model_logic import update_no_hand_banner, update_ui_on_prediction_phrases
import joblib
from GUI.gui_utils import show_completion_popup

# Constants
MODEL_FRAMES = 15
MIN_LENGTH_FRAMES = 5

# Cargar el modelo dinámico TFLite
phrases_model_path = os.path.join(phrases_model_keras_path, phrases_model_lite_name)
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
scaler_path = os.path.join(phrases_model_keras_path, 'scaler.save')
scaler = joblib.load(scaler_path)


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


def load_phrase_data(actual_phrase):
    """Carga la información de una frase específica desde un archivo JSON."""
    DATA_LETTERS_JSON_PATH = "GUI/Assets/Phrases_Information/phrases_data.json"
    with open(DATA_LETTERS_JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data.get(actual_phrase, None)


def start_phrases_complete_recognition_cycle(window, video_label, progress_label, current_letter_index=0):
    """Inicia el ciclo de reconocimiento de señas con el modelo TFLite y actualiza la interfaz."""
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

        # Verificar si se han completado todas las frases
        if current_letter_index >= len(PHRASES):
            show_completion_popup(window, progress_label, lambda: restart_lesson(window, video_label, progress_label))
            return

        actual_phrase = PHRASES[current_letter_index]
        phrase_data = load_phrase_data(actual_phrase)
        movement_type = phrase_data["movement"]

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

            if not transitioning:
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

                        # Realizar predicción con el modelo TFLite
                        interpreter_dynamic.resize_tensor_input(input_details_dynamic[0]['index'],
                                                                [1, MODEL_FRAMES, num_keypoints])
                        interpreter_dynamic.allocate_tensors()
                        interpreter_dynamic.set_tensor(input_details_dynamic[0]['index'], kp_scaled[np.newaxis, ...])
                        interpreter_dynamic.invoke()
                        res = interpreter_dynamic.get_tensor(output_details_dynamic[0]['index'])[0]

                        pred_index = np.argmax(res)
                        confidence = res[pred_index] * 100

                        if confidence > 50:  # threshold = 0.5
                            predicted_word = word_ids[pred_index]
                            general_word = phrases_to_text.get(predicted_word, predicted_word)
                            actual_phrase_lower = actual_phrase.lower()

                            print(f"\nPredicción detectada:")
                            print(f"- Palabra predicha (original): {predicted_word}")
                            print(f"- Palabra predicha (general): {general_word}")
                            print(f"- Confianza: {confidence:.2f}%")
                            print(f"- Índice del modelo: {pred_index}")

                            if general_word == actual_phrase_lower and not transitioning:
                                print("¡CORRECTO! La seña coincide con el objetivo")
                                transitioning = True
                                update_ui_on_prediction_phrases(window, actual_phrase, actual_phrase)
                                correct_count += 1
                                progress_label.config(text=f"{correct_count}/{len(PHRASES)}")
                                window.after(2000, transition_to_next_letter)
                            else:
                                print(f"INCORRECTO: La seña no coincide con el objetivo")
                                print(f"- Esperado: {actual_phrase_lower}")
                                print(f"- Recibido: {general_word}")
                                update_ui_on_prediction_phrases(window, "incorrect", actual_phrase)
                        else:
                            print(f"\nBaja confianza en la predicción:")
                            print(f"- Confianza: {confidence:.2f}%")
                            print(f"- Umbral requerido: 50%")
                            update_ui_on_prediction_phrases(window, "low_confidence", actual_phrase)

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
        if current_letter_index < len(PHRASES):
            last_correct_prediction = PHRASES[current_letter_index]
        transitioning = False

    update_frame()

    def on_closing():
        cap.release()
        holistic_model.close()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_closing)


def next_letter(window, video_label, current_letter_index, progress_label):
    current_letter_index += 1
    if current_letter_index >= len(PHRASES):
        # Al terminar la última frase, se mostrará el popup en update_frame
        # y luego se llamará a restart_lesson
        return current_letter_index

    actual_phrase = PHRASES[current_letter_index]
    phrase_data = load_phrase_data(actual_phrase)
    if phrase_data:
        update_icon_letter(window, phrase_data["icon_path"])
    return current_letter_index


def restart_lesson(window, video_label, progress_label):
    """Reinicia la lección desde la primera frase."""
    if isinstance(progress_label, Label):
        progress_label.config(text="0/7")
    start_phrases_complete_recognition_cycle(window, video_label, progress_label)
