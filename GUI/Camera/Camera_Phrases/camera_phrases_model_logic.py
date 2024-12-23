import os
import cv2
import json
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from tkinter import Label, messagebox, PhotoImage
import PIL.Image, PIL.ImageTk
from Utils.paths import phrases_model_json_data_path, dynamic_model_keras_path
from Utils.config import dynamic_model_lite_name, id_camera, phrases_to_text
from GUI.Camera.Camera_Letters.camera_letters_model_logic import relative_to_assets_camera
import time
import joblib
import tensorflow as tf

# Constants
MODEL_FRAMES = 15
MIN_LENGTH_FRAMES = 5

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

def update_no_hand_banner(window, hand_detected):
    canvas = window.children['!canvas']

    if not hand_detected:
        if not hasattr(window, 'no_hand_banner'):
            banner_image_filename = "banner_no_mano.png"
            banner_image_path = relative_to_assets_camera(banner_image_filename)
            banner_image = PhotoImage(file=banner_image_path)
            banner = canvas.create_image(925.0, 668.5948486328125, image=banner_image)
            window.no_hand_banner_image = banner_image
            window.no_hand_banner = banner
    else:
        if hasattr(window, 'no_hand_banner'):
            canvas.delete(window.no_hand_banner)
            del window.no_hand_banner
            del window.no_hand_banner_image

def update_ui_on_prediction_phrases(window, predicted_label, actual_phrase, duration=2000):
    canvas = window.children['!canvas']  # Obtén el canvas de la ventana

    if hasattr(window, 'prediction_banner'):
        canvas.delete(window.prediction_banner)
        del window.prediction_banner
        del window.prediction_banner_image

    if predicted_label == actual_phrase:
        banner_image_filename = "banner_bien.png"
    elif predicted_label == "low_confidence":
        banner_image_filename = "banner_intento.png"
    else:
        banner_image_filename = "banner_intento.png"

    banner_image_path = relative_to_assets_camera(banner_image_filename)
    banner_image = PhotoImage(file=banner_image_path)
    banner = canvas.create_image(925.0, 668.5948486328125, image=banner_image)
    window.prediction_banner_image = banner_image
    window.prediction_banner = banner

    def remove_prediction_banner():
        if hasattr(window, 'prediction_banner'):
            canvas.delete(window.prediction_banner)
            del window.prediction_banner
            del window.prediction_banner_image

    window.after(duration, remove_prediction_banner)

def evaluate_phrases_model(video_label, window, actual_phrase, threshold=0.5, margin_frame=1, delay_frames=3, debug=True):
    kp_seq, sentence = [], []
    actual_phrase_lower = actual_phrase.lower()

    if debug:
        print("\n=== Iniciando reconocimiento de señas ===")
        print(f"Frase objetivo (original): {actual_phrase}")
        print(f"Frase objetivo (minúsculas para comparación): {actual_phrase_lower}")
        print("========================================\n")

    # Cargar IDs de palabras
    with open(phrases_model_json_data_path, 'r') as json_file:
        data = json.load(json_file)
        word_ids = data.get('word_ids')
        if not word_ids:
            raise ValueError("[ERROR] No se encontraron identificadores de palabras en el archivo JSON.")

    # Cargar el modelo TFLite
    model_path = os.path.join(dynamic_model_keras_path, dynamic_model_lite_name)
    interpreter_dynamic = tf.lite.Interpreter(model_path=model_path)
    interpreter_dynamic.allocate_tensors()
    input_details_dynamic = interpreter_dynamic.get_input_details()
    output_details_dynamic = interpreter_dynamic.get_output_details()

    # Cargar el scaler
    scaler_path = os.path.join(dynamic_model_keras_path, 'scaler.save')
    scaler = joblib.load(scaler_path)

    # Inicializar la cámara y Holistic
    video = cv2.VideoCapture(id_camera)
    if not video.isOpened():
        raise RuntimeError("[ERROR] No se pudo abrir la cámara. Revisa el índice o la conexión de la cámara.")

    if debug:
        print("[DEBUG] Cámara abierta correctamente.")

    holistic_model = Holistic()
    if debug:
        print("[DEBUG] Modelo Holistic inicializado.")

    count_frame = 0
    fix_frames = 0
    recording = False
    photo = None
    running = True
    after_id = None

    def update_frame():
        nonlocal count_frame, fix_frames, recording, kp_seq, sentence, photo, running, after_id

        if not running:
            return

        ret, frame = video.read()
        if not ret:
            if debug:
                print("[DEBUG] No se pudo leer el frame de la cámara. Deteniendo actualización de video.")
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic_model.process(image)

        hand_detected = results.left_hand_landmarks or results.right_hand_landmarks
        update_no_hand_banner(window, hand_detected)

        if not hand_detected:
            recording = False
            count_frame = 0
            fix_frames = 0
            kp_seq = []
        else:
            recording = True
            count_frame += 1
            if count_frame > margin_frame:
                pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                                results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
                face = np.array([[res.x, res.y, res.z] for res in
                                results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
                left_hand = np.array([[res.x, res.y, res.z] for res in
                                    results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
                right_hand = np.array([[res.x, res.y, res.z] for res in
                                    results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
                kp_frame = np.concatenate([pose, face, left_hand, right_hand])
                kp_seq.append(kp_frame)

            if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                fix_frames += 1
                if fix_frames >= delay_frames:
                    recording = False
                    if len(kp_seq) > (margin_frame + delay_frames):
                        kp_seq = kp_seq[:-(margin_frame + delay_frames)]

                    kp_normalized = normalize_keypoints(kp_seq, MODEL_FRAMES)
                    kp_normalized = np.array(kp_normalized)
                    num_frames, num_keypoints = kp_normalized.shape

                    kp_flat = kp_normalized.reshape(-1, num_keypoints)
                    kp_scaled = scaler.transform(kp_flat)
                    kp_scaled = kp_scaled.reshape(num_frames, num_keypoints)
                    kp_input = kp_scaled.reshape(1, MODEL_FRAMES, num_keypoints).astype(np.float32)

                    interpreter_dynamic.set_tensor(input_details_dynamic[0]['index'], kp_input)
                    interpreter_dynamic.invoke()
                    res = interpreter_dynamic.get_tensor(output_details_dynamic[0]['index'])[0]

                    pred_index = np.argmax(res)
                    confidence = res[pred_index] * 100

                    if debug:
                        print(f"[DEBUG] Predicción: Índice={pred_index}, Confianza={confidence:.2f}%")

                    if confidence > threshold * 100:
                        predicted_word = word_ids[pred_index]
                        general_word = phrases_to_text.get(predicted_word, predicted_word)

                        if debug:
                            print(f"[DEBUG] Palabra detectada: {predicted_word} ({general_word}) con confianza {confidence:.2f}%")

                        if general_word == actual_phrase_lower:
                            update_ui_on_prediction_phrases(window, general_word, actual_phrase_lower)
                            if debug:
                                print("[DEBUG] ¡CORRECTO! La seña coincide con el objetivo.")
                        else:
                            update_ui_on_prediction_phrases(window, "incorrect", actual_phrase_lower)
                            if debug:
                                print(f"[DEBUG] INCORRECTO: Esperado={actual_phrase_lower}, Recibido={general_word}")
                    else:
                        update_ui_on_prediction_phrases(window, "low_confidence", actual_phrase_lower)
                        if debug:
                            print("[DEBUG] Confianza baja, no se supera el umbral.")

                    count_frame = 0
                    fix_frames = 0
                    kp_seq = []
                else:
                    recording = True

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = PIL.Image.fromarray(frame_rgb)
        frame_pil = frame_pil.resize((784, 576))
        photo = PIL.ImageTk.PhotoImage(image=frame_pil)
        video_label.configure(image=photo)
        video_label.image = photo

        after_id = window.after(10, update_frame)

    def on_closing():
        nonlocal running, after_id
        running = False
        if debug:
            print("[DEBUG] on_closing llamado, liberando recursos.")
        if after_id is not None:
            window.after_cancel(after_id)
            after_id = None
        if video.isOpened():
            video.release()
            print("[DEBUG] Cámara liberada correctamente.")
        holistic_model.close()
        print("[DEBUG] Modelo Holistic cerrado.")
        if window.winfo_exists():
            window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_closing)
    update_frame()

    return sentence