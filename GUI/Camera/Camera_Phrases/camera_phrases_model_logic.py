import os
import cv2
import json
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from tkinter import Label, messagebox, PhotoImage
import PIL.Image, PIL.ImageTk
from Utils.paths import phrases_model_json_data_path, phrases_model_keras_path
from Utils.config import phrases_model_lite_name, id_camera, phrases_to_text
from GUI.Camera.Camera_Letters.camera_letters_model_logic import relative_to_assets_camera
import time
import joblib  # Asegúrate de importar joblib si usas scaler
import tensorflow as tf

# Constants
MODEL_FRAMES = 15
MIN_LENGTH_FRAMES = 5

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

def update_no_hand_banner(window, hand_detected):
    """
    Muestra u oculta el banner de 'No se detecta ninguna mano' según si se detecta una mano o no.
    """
    canvas = window.children['!canvas']  # Obtén el canvas de la ventana

    if not hand_detected:
        # Si no se detecta ninguna mano, mostrar el banner si no está ya mostrado
        if not hasattr(window, 'no_hand_banner'):
            banner_image_filename = "banner_no_mano.png"
            banner_image_path = relative_to_assets_camera(banner_image_filename)
            banner_image = PhotoImage(file=banner_image_path)
            banner = canvas.create_image(925.0, 668.5948486328125, image=banner_image)

            # Guardar referencias para evitar que se eliminen
            window.no_hand_banner_image = banner_image
            window.no_hand_banner = banner
    else:
        # Si se detecta una mano, eliminar el banner si existe
        if hasattr(window, 'no_hand_banner'):
            canvas.delete(window.no_hand_banner)
            del window.no_hand_banner
            del window.no_hand_banner_image

def update_ui_on_prediction_phrases(window, predicted_label, actual_phrase, duration=2000):
    """
    Actualiza la interfaz dependiendo de la predicción de la seña y muestra un banner temporal.

    window -- Ventana principal de Tkinter
    predicted_label -- Etiqueta predicha por el modelo
    actual_phrase -- Frase objetivo para comparar con la predicción (en minúsculas)
    duration -- Duración en milisegundos que el banner será visible
    """
    canvas = window.children['!canvas']  # Obtén el canvas de la ventana

    # Eliminar cualquier banner de predicción existente
    if hasattr(window, 'prediction_banner'):
        canvas.delete(window.prediction_banner)
        del window.prediction_banner
        del window.prediction_banner_image

    # Determinar qué imagen de banner mostrar
    if predicted_label == actual_phrase:
        banner_image_filename = "banner_bien.png"
    elif predicted_label == "low_confidence":
        banner_image_filename = "banner_intento.png"
    else:
        banner_image_filename = "banner_intento.png"

    # Cargar la imagen del banner
    banner_image_path = relative_to_assets_camera(banner_image_filename)
    banner_image = PhotoImage(file=banner_image_path)
    banner = canvas.create_image(925.0, 668.5948486328125, image=banner_image)

    # Guardar referencias para evitar que se eliminen
    window.prediction_banner_image = banner_image
    window.prediction_banner = banner

    # Función interna para eliminar el banner después del retraso
    def remove_prediction_banner():
        if hasattr(window, 'prediction_banner'):
            canvas.delete(window.prediction_banner)
            del window.prediction_banner
            del window.prediction_banner_image

    # Programar la eliminación del banner después de 'duration' milisegundos
    window.after(duration, remove_prediction_banner)

def evaluate_phrases_model(video_label, window, actual_phrase, threshold=0.5, margin_frame=1, delay_frames=3):
    """
    Evalúa el modelo LSTM para traducir señas dinámicas en una interfaz Tkinter,
    utilizando un modelo TFLite en lugar de un modelo Keras.

    Parámetros:
    video_label -- Label de Tkinter donde se mostrará el video
    window -- Ventana principal de Tkinter
    actual_phrase -- Frase objetivo para comparar con la predicción (en mayúsculas)
    threshold -- Umbral mínimo de confianza para aceptar una predicción
    margin_frame -- Frames iniciales ignorados antes de capturar
    delay_frames -- Frames de retardo antes de detener la captura
    """
    kp_seq, sentence = [], []

    # Convertir la frase objetivo a minúsculas para comparación
    actual_phrase_lower = actual_phrase.lower()

    print("\n=== Iniciando reconocimiento de señas ===")
    print(f"Frase objetivo (original): {actual_phrase}")
    print(f"Frase objetivo (minúsculas para comparación): {actual_phrase_lower}")
    print("========================================\n")

    # Cargar identificadores de palabras desde el archivo JSON
    with open(phrases_model_json_data_path, 'r') as json_file:
        data = json.load(json_file)
        word_ids = data.get('word_ids')
        if not word_ids:
            raise ValueError("[ERROR] No se encontraron identificadores de palabras en el archivo JSON.")

    # Cargar el modelo TFLite
    model_path = os.path.join(phrases_model_keras_path, phrases_model_lite_name)
    interpreter_dynamic = tf.lite.Interpreter(model_path=model_path)
    interpreter_dynamic.allocate_tensors()
    input_details_dynamic = interpreter_dynamic.get_input_details()
    output_details_dynamic = interpreter_dynamic.get_output_details()

    count_frame = 0
    fix_frames = 0
    recording = False

    # Variable para almacenar la última imagen procesada
    photo = None

    # Cargar el scaler
    scaler_path = os.path.join(phrases_model_keras_path, 'scaler.save')
    scaler = joblib.load(scaler_path)

    # Inicializar la cámara y el modelo Holistic
    video = cv2.VideoCapture(id_camera)
    holistic_model = Holistic()

    def update_frame():
        nonlocal count_frame, fix_frames, recording, kp_seq, sentence, photo

        ret, frame = video.read()
        if ret:

            # Procesar el frame con MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic_model.process(image)

            # Detectar manos
            hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

            # Actualizar el banner de 'No se detecta ninguna mano'
            update_no_hand_banner(window, hand_detected)

            if not hand_detected:
                # Reiniciar variables si es necesario
                recording = False
                count_frame = 0
                fix_frames = 0
                kp_seq = []
            else:
                # Captura los frames mientras haya manos detectadas
                recording = True
                count_frame += 1
                if count_frame > margin_frame:
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
                    kp_seq.append(kp_frame)

                # Verificar si se ha completado la secuencia para predecir
                if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                    fix_frames += 1
                    if fix_frames >= delay_frames:
                        recording = False
                        kp_seq = kp_seq[: - (margin_frame + delay_frames)]
                        kp_normalized = normalize_keypoints(kp_seq, int(MODEL_FRAMES))
                        kp_normalized = np.array(kp_normalized)
                        num_frames, num_keypoints = kp_normalized.shape

                        # Aplicar el scaler
                        kp_flat = kp_normalized.reshape(-1, num_keypoints)
                        kp_scaled = scaler.transform(kp_flat)
                        kp_scaled = kp_scaled.reshape(num_frames, num_keypoints)

                        # Realizar predicción con el modelo TFLite
                        # Ajustar la forma de entrada según el modelo (ej: [1, 15, num_keypoints])
                        kp_input = kp_scaled.reshape(1, MODEL_FRAMES, num_keypoints).astype(np.float32)

                        interpreter_dynamic.set_tensor(input_details_dynamic[0]['index'], kp_input)
                        interpreter_dynamic.invoke()
                        res = interpreter_dynamic.get_tensor(output_details_dynamic[0]['index'])[0]

                        # Obtener el índice y valor de la predicción máxima
                        pred_index = np.argmax(res)
                        confidence = res[pred_index] * 100

                        if confidence > threshold * 100:
                            # Obtener la predicción original
                            predicted_word = word_ids[pred_index]

                            # Mapear la predicción a su forma general utilizando phrases_to_text
                            general_word = phrases_to_text.get(predicted_word, predicted_word)

                            print(f"\nPredicción detectada:")
                            print(f"- Palabra predicha (original): {predicted_word}")
                            print(f"- Palabra predicha (general): {general_word}")
                            print(f"- Confianza: {confidence:.2f}%")
                            print(f"- Índice del modelo: {pred_index}")

                            # Actualizar UI basado en la predicción general
                            if general_word == actual_phrase_lower:
                                print("¡CORRECTO! La seña coincide con el objetivo")
                                update_ui_on_prediction_phrases(window, general_word, actual_phrase_lower)
                            else:
                                print(f"INCORRECTO: La seña no coincide con el objetivo")
                                print(f"- Esperado: {actual_phrase_lower}")
                                print(f"- Recibido: {general_word}")
                                update_ui_on_prediction_phrases(window, "incorrect", actual_phrase_lower)
                        else:
                            print(f"\nBaja confianza en la predicción:")
                            print(f"- Confianza: {confidence:.2f}%")
                            print(f"- Umbral requerido: {threshold * 100}%")
                            update_ui_on_prediction_phrases(window, "low_confidence", actual_phrase_lower)

                        # Resetea los estados
                        count_frame = 0
                        fix_frames = 0
                        kp_seq = []
                    else:
                        recording = True  # Continuar grabando hasta alcanzar delay_frames

            # Convertir el frame para mostrarlo en Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = PIL.Image.fromarray(frame)
            frame = frame.resize((784, 576))  # Ajustar al tamaño del video_label
            photo = PIL.ImageTk.PhotoImage(image=frame)
            video_label.configure(image=photo)
            video_label.image = photo  # Mantener referencia para evitar que sea recolectada

            # Programar la siguiente actualización
            window.after(10, update_frame)
        else:
            # Si no se pudo leer el frame, cerrar el video y salir
            video.release()
            holistic_model.close()
            window.destroy()

    # Inicializar la cámara y el modelo Holistic
    video = cv2.VideoCapture(id_camera)
    holistic_model = Holistic()

    # Iniciar la actualización de frames
    update_frame()

    # Función para limpiar recursos cuando se cierre la ventana
    def on_closing():
        video.release()
        holistic_model.close()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_closing)

    return sentence
