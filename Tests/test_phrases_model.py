import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Model.model_utils import *
from Utils.paths import phrases_model_json_data_path, generated_models_path


def interpolate_keypoints(keypoints, target_length=15):
    """
    Interpola los puntos clave para ajustarlos al tamaño deseado.

    Parámetros:
    keypoints -- Lista de puntos clave a interpolar.
    target_length -- Longitud objetivo para los puntos clave.

    Retorna:
    - Lista de puntos clave interpolados.
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

    Parámetros:
    keypoints -- Lista de puntos clave.
    target_length -- Longitud objetivo de los puntos clave.

    Retorna:
    - Lista de puntos clave normalizados.
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


def evaluate_model(src=None, threshold=0.8, margin_frame=1, delay_frames=3):
    """
    Evalúa el modelo LSTM para traducir señas dinámicas.

    Parámetros:
    src -- Fuente del video (cámara por defecto).
    threshold -- Umbral mínimo de confianza para aceptar una predicción.
    margin_frame -- Número de frames iniciales a ignorar antes de comenzar a capturar.
    delay_frames -- Frames de retardo antes de detener la captura tras no detectar manos.

    Retorna:
    - Lista con las palabras detectadas como resultado de las señas.
    """
    kp_seq, sentence = [], []
    word_ids = get_word_ids(phrases_model_json_data_path)
    model = load_model(os.path.join(generated_models_path, "phrases_model.keras"))
    count_frame = 0
    fix_frames = 0
    recording = False

    # Inicializa el modelo de MediaPipe Holistic
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(src or 1)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Procesa el frame y obtiene los puntos clave
            results = mediapipe_detection(frame, holistic_model)

            # Captura los frames mientras haya manos detectadas
            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    kp_frame = extract_keypoints(results)
                    kp_seq.append(kp_frame)
            else:
                # Normaliza y predice cuando se termina la captura
                if count_frame >= 5 + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    kp_seq = kp_seq[: - (margin_frame + delay_frames)]
                    kp_normalized = normalize_keypoints(kp_seq, int(15))
                    res = model.predict(np.expand_dims(kp_normalized, axis=0))[0]

                    # Selecciona la palabra con mayor probabilidad si supera el umbral
                    print(np.argmax(res), f"({res[np.argmax(res)] * 100:.2f}%)")
                    if res[np.argmax(res)] > threshold:
                        word_id = word_ids[np.argmax(res)].split('-')[0]
                        sentence.insert(0, words_text.get(word_id))

                # Resetea los estados
                recording = False
                fix_frames = 0
                count_frame = 0
                kp_seq = []

            # Muestra la traducción en pantalla si no es una fuente de video específica
            if not src:
                cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
                # SHOW IMAGE PARAMETERS
                FONT = cv2.FONT_HERSHEY_PLAIN
                FONT_SIZE = 1.5
                FONT_POS = (5, 30)
                cv2.putText(frame, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))

                draw_keypoints(frame, results)
                cv2.imshow('Traductor LSP', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        # Libera recursos
        video.release()
        cv2.destroyAllWindows()
        return sentence


if __name__ == "__main__":
    """
    Script principal para evaluar un modelo entrenado.
    """
    # Llama a la función de evaluación
    resultado = evaluate_model()
    print(f"Frase traducida: {' '.join(resultado)}")
