import os
import cv2
import json
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from Model.text_to_speech import initialize_tts, speak_text
from typing import NamedTuple
from Utils.paths import phrases_model_json_data_path, phrases_model_keras_path
from Utils.config import phrases_model_keras_name, id_camera

# Constantes
MODEL_FRAMES = 15
MIN_LENGTH_FRAMES = 5

phrases_to_text = {
    "como_estas": "COMO ESTAS",
    "por_favor": "POR FAVOR",
    "hola_izq": "HOLA",
    "hola_der": "HOLA",
    "de_nada_izq": "DE NADA",
    "de_nada_der": "DE NADA",
    "adios_izq": "ADIOS",
    "adios_der": "ADIOS",
    "cuidate": "CUIDATE",
    "mas_o_menos_izq": "MAS O MENOS",
    "mas_o_menos_der": "MAS O MENOS",
    "gracias_izq": "GRACIAS",
    "gracias_der": "GRACIAS",
    "j_izq": "J",
    "j_der": "J",
    "q_izq": "Q",
    "q_der": "Q",
    "x_izq": "X",
    "x_der": "X",
    "z_izq": "Z",
    "z_der": "Z"
}

def pad_keypoints_sequence(kp_seq, target_length=15):
    """
    Ajusta la secuencia de keypoints al tamaño objetivo mediante truncamiento o relleno.
    """
    current_length = len(kp_seq)
    if current_length >= target_length:
        # Trunca la secuencia a los primeros `target_length` keypoints
        return kp_seq[:target_length]
    else:
        # Repite el último keypoint hasta alcanzar el número deseado
        last_frame = kp_seq[-1]
        padding = [last_frame.copy() for _ in range(target_length - current_length)]
        return kp_seq + padding

def evaluate_model(src=None, threshold=0.7, margin_frame=2, delay_frames=3):
    """
    Evalúa el modelo LSTM para traducir señas dinámicas.
    """
    kp_seq, sentence = [], []

    # Cargar el scaler
    import joblib
    scaler_path = os.path.join(phrases_model_keras_path, 'scaler.save')
    scaler = joblib.load(scaler_path)

    # Cargar identificadores de palabras desde el archivo JSON
    with open(phrases_model_json_data_path, 'r') as json_file:
        data = json.load(json_file)
        word_ids = data.get('word_ids')
        if not word_ids:
            raise ValueError("[ERROR] No se encontraron identificadores de palabras en el archivo JSON.")

    # Inicializar el motor de texto a voz
    tts_engine = initialize_tts()

    # Cargar el modelo entrenado
    model = load_model(os.path.join(phrases_model_keras_path, phrases_model_keras_name))
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
            image.flags.writeable = False
            results = holistic_model.process(image)

            # Detectar manos
            hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

            # Captura los frames mientras haya manos detectadas
            if hand_detected or recording:
                recording = False
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
            else:
                # Normaliza y predice cuando se termina la captura
                if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue

                    kp_seq = kp_seq[: -delay_frames] if delay_frames > 0 else kp_seq

                    # Utiliza la nueva función de padding
                    kp_padded = pad_keypoints_sequence(kp_seq, target_length=MODEL_FRAMES)

                    kp_padded = np.array(kp_padded)
                    num_frames, num_keypoints = kp_padded.shape

                    # Aplicar el scaler
                    kp_flat = kp_padded.reshape(-1, num_keypoints)
                    kp_scaled = scaler.transform(kp_flat)
                    kp_scaled = kp_scaled.reshape(num_frames, num_keypoints)

                    res = model.predict(np.expand_dims(kp_scaled, axis=0))[0]

                    # Selecciona la palabra con mayor probabilidad si supera el umbral
                    print(np.argmax(res), f"({res[np.argmax(res)] * 100:.2f}%)")

                    if res[np.argmax(res)] > threshold:
                        word_id = word_ids[np.argmax(res)].split('-')[0]
                        sentence.insert(0, word_id)

                        # Obtiene la traducción limpia desde `phrases_to_text`
                        spoken_text = phrases_to_text.get(word_id, word_id)

                        print(f"Frase traducida:", spoken_text)

                        # Reproducir texto a voz
                        # speak_text(tts_engine, spoken_text)

                # Resetea los estados
                recording = False
                fix_frames = 0
                count_frame = 0
                kp_seq = []

            # Muestra la traducción en pantalla si no es una fuente de video específica
            if not src:
                cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
                cv2.putText(frame, ' | '.join(sentence), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                cv2.imshow('Traductor Lengua de Señas Mexicana', frame)
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