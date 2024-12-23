import os
import cv2
import json
import numpy as np
import traceback
from mediapipe.python.solutions.holistic import Holistic
from tensorflow.lite.python.interpreter import Interpreter
from Utils.paths import dynamic_model_json_data_path, dynamic_model_keras_path
from Utils.config import id_camera, phrases_to_text, dynamic_model_lite_name

# Constantes
MODEL_FRAMES = 15
MIN_LENGTH_FRAMES = 5
TFLITE_MODEL_PATH = os.path.join(dynamic_model_keras_path, dynamic_model_lite_name)

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

def evaluate_model_tflite(src=None, threshold=0.7, margin_frame=2, delay_frames=3, debug=True):
    kp_seq, sentence = [], []

    # Cargar el scaler
    import joblib
    scaler_path = os.path.join(dynamic_model_keras_path, 'scaler.save')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No se encontró el scaler en {scaler_path}")
    scaler = joblib.load(scaler_path)
    if debug: print("[DEBUG] Scaler cargado correctamente.")

    # Cargar identificadores de palabras desde el JSON
    if not os.path.exists(dynamic_model_json_data_path):
        raise FileNotFoundError(f"No se encontró el archivo JSON: {dynamic_model_json_data_path}")

    with open(dynamic_model_json_data_path, 'r') as json_file:
        data = json.load(json_file)
        word_ids = data.get('word_ids', [])
        if not word_ids:
            raise ValueError("[ERROR] No se encontraron identificadores de palabras en el archivo JSON.")
    if debug: print(f"[DEBUG] Se cargaron {len(word_ids)} palabras: {word_ids}")

    # Cargar el modelo TFLite
    if not os.path.exists(TFLITE_MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo TFLite en {TFLITE_MODEL_PATH}")

    interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    if debug: print("[DEBUG] Modelo TFLite cargado y tensores asignados.")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if debug:
        print(f"[DEBUG] Detalles de entrada del modelo TFLite: {input_details}")
        print(f"[DEBUG] Detalles de salida del modelo TFLite: {output_details}")

    count_frame = 0
    fix_frames = 0
    recording = False

    # Abrir la cámara
    if debug: print("[DEBUG] Intentando abrir la cámara...")
    video = cv2.VideoCapture(id_camera if src is None else src)
    if not video.isOpened():
        raise RuntimeError("[ERROR] No se pudo abrir la cámara, verifica el índice o la conexión.")

    if debug: print("[DEBUG] Cámara abierta correctamente.")

    try:
        with Holistic() as holistic_model:
            if debug: print("[DEBUG] Holistic model inicializado.")

            cv2.namedWindow('Evaluación de modelo TFLite', cv2.WINDOW_NORMAL)
            while True:
                ret, frame = video.read()
                if not ret or frame is None:
                    if debug: print("[DEBUG] No se recibió frame de la cámara, saliendo del bucle...")
                    break

                # Procesar el frame con MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic_model.process(image)

                hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

                # Lógica de grabación
                if hand_detected:
                    if debug: print("[DEBUG] Manos detectadas, iniciando/continuando grabación de frames.")
                    recording = True
                    count_frame += 1
                    if count_frame > margin_frame:
                        # Extraer keypoints
                        if debug: print("[DEBUG] Extrayendo keypoints de pose, cara y manos.")
                        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
                        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
                        left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
                        right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

                        kp_frame = np.concatenate([pose, face, left_hand, right_hand])
                        kp_seq.append(kp_frame)

                else:
                    if recording:
                        # Si se estaba grabando y ya no hay manos, intentar predecir
                        if debug: print("[DEBUG] Finalizando grabación, intentando predecir.")
                        if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                            fix_frames += 1
                            if fix_frames >= delay_frames:
                                kp_seq = kp_seq[:-delay_frames] if len(kp_seq) >= delay_frames else kp_seq
                                if debug: print("[DEBUG] Normalizando keypoints para predecir.")
                                kp_normalized = normalize_keypoints(kp_seq, MODEL_FRAMES)
                                kp_flat = np.array(kp_normalized).reshape(-1, kp_normalized[0].shape[-1])
                                kp_scaled = scaler.transform(kp_flat)
                                kp_scaled = kp_scaled.reshape(1, MODEL_FRAMES, -1)

                                if debug: print("[DEBUG] Realizando inferencia TFLite.")
                                interpreter.set_tensor(input_details[0]['index'], kp_scaled.astype(np.float32))
                                interpreter.invoke()
                                res = interpreter.get_tensor(output_details[0]['index'])[0]

                                pred_conf = np.max(res)
                                pred_class = np.argmax(res)
                                if pred_conf > threshold:
                                    word_id = word_ids[pred_class]
                                    sentence.insert(0, word_id)
                                    print(f"[INFO] Palabra detectada: {word_id} ({pred_conf * 100:.2f}%)")
                                else:
                                    if debug:
                                        print(f"[DEBUG] Confianza: {pred_conf:.2f}, menor que umbral {threshold}. Ninguna palabra detectada.")

                                # Resetear variables
                                recording = False
                                fix_frames = 0
                                count_frame = 0
                                kp_seq = []
                            else:
                                if debug: print("[DEBUG] Esperando frames de retardo...")
                        else:
                            if debug: print("[DEBUG] No se capturaron suficientes frames para predecir.")
                            recording = False
                            fix_frames = 0
                            count_frame = 0
                            kp_seq = []
                    else:
                        # No se detectan manos y no se estaba grabando, continuar
                        pass

                cv2.imshow('Evaluación de modelo TFLite', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    if debug: print("[DEBUG] 'q' presionado, saliendo del bucle.")
                    break

    except Exception as e:
        print("[ERROR] Ocurrió una excepción durante la ejecución:")
        traceback.print_exc()
    finally:
        if video.isOpened():
            video.release()
        cv2.destroyAllWindows()
        if debug: print("[DEBUG] Recursos liberados correctamente.")

    return sentence

if __name__ == "__main__":
    resultado = evaluate_model_tflite(debug=True)
    print("[INFO] Resultado final:", resultado)
