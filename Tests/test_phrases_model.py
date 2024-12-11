import os
import cv2
import json
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from tensorflow.lite.python.interpreter import Interpreter
from Model.text_to_speech import initialize_tts, speak_text
from Utils.paths import phrases_model_json_data_path, phrases_model_keras_path
from Utils.config import id_camera, phrases_to_text

# Constantes
MODEL_FRAMES = 15
MIN_LENGTH_FRAMES = 5
TFLITE_MODEL_PATH = os.path.join(phrases_model_keras_path, "phrases_model.tflite")


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


def evaluate_model_tflite(src=None, threshold=0.7, margin_frame=2, delay_frames=3):
    """
    Evalúa el modelo TFLite para traducir señas dinámicas.

    Parámetros:
    src -- Fuente del video (cámara por defecto).
    threshold -- Umbral mínimo de confianza para aceptar una predicción.
    margin_frame -- Frames iniciales ignorados antes de capturar.
    delay_frames -- Frames de retardo antes de detener la captura.

    Retorna:
    - Lista de palabras detectadas como resultado de las señas.
    """
    kp_seq, sentence = [], []

    # Cargar el scaler
    import joblib
    scaler_path = os.path.join(phrases_model_keras_path, 'scaler.save')
    scaler = joblib.load(scaler_path)

    # Cargar identificadores de palabras desde el archivo JSON
    with open(phrases_model_json_data_path, 'r') as json_file:
        data = json.load(json_file)
        word_ids = data.get('word_ids', [])
        if not word_ids:
            raise ValueError("[ERROR] No se encontraron identificadores de palabras en el archivo JSON.")

    # Inicializar el motor de texto a voz
    tts_engine = initialize_tts()

    # Cargar el modelo TFLite
    interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    # Obtener detalles de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    count_frame = 0
    fix_frames = 0
    recording = False

    # Inicializa el modelo de MediaPipe Holistic
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(id_camera)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Procesar el frame con MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic_model.process(image)

            # Detectar manos
            hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

            if hand_detected or recording:
                count_frame += 1
                if count_frame > margin_frame:
                    # Extraer keypoints
                    pose = np.zeros(33 * 4)
                    if results.pose_landmarks:
                        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()

                    face = np.zeros(468 * 3)
                    if results.face_landmarks:
                        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()

                    left_hand = np.zeros(21 * 3)
                    if results.left_hand_landmarks:
                        left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()

                    right_hand = np.zeros(21 * 3)
                    if results.right_hand_landmarks:
                        right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()

                    kp_frame = np.concatenate([pose, face, left_hand, right_hand])
                    kp_seq.append(kp_frame)

            else:
                if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        continue

                    kp_seq = kp_seq[: -delay_frames]
                    kp_normalized = normalize_keypoints(kp_seq, MODEL_FRAMES)
                    kp_flat = np.array(kp_normalized).reshape(-1, kp_normalized[0].shape[-1])
                    kp_scaled = scaler.transform(kp_flat)
                    kp_scaled = kp_scaled.reshape(1, MODEL_FRAMES, -1)

                    # Realizar predicción con TFLite
                    interpreter.set_tensor(input_details[0]['index'], kp_scaled.astype(np.float32))
                    interpreter.invoke()
                    res = interpreter.get_tensor(output_details[0]['index'])[0]

                    if res[np.argmax(res)] > threshold:
                        word_id = word_ids[np.argmax(res)]
                        sentence.insert(0, word_id)
                        print(f"[INFO] Palabra detectada: {word_id} ({res[np.argmax(res)] * 100:.2f}%)")
                    else:
                        print(f"[INFO] Ninguna predicción supera el umbral ({threshold}).")

                # Reset
                recording = False
                fix_frames = 0
                count_frame = 0
                kp_seq = []

            cv2.imshow('Evaluación de modelo TFLite', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    Script principal para evaluar el modelo TFLite.
    """
    resultado = evaluate_model_tflite()
