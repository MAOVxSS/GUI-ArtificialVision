import os
import numpy as np
import cv2
from mediapipe.python.solutions.hands import Hands, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from tensorflow.keras.models import load_model

# Rutas y variables
from Utils.paths import dynamic_model_converted_data_path, generated_models_path
from Utils.config import id_camera, dynamic_model_name

# Funciones
from Model.model_utils import mediapipe_detection


# Función para probar el modelo dinámico
def evaluate_dynamic_model(model, threshold=0.7):
    # Inicialización de variables
    max_length_frames = 15  # Cantidad de frames maxima
    keypoint_sequence, sentence = [], []
    actions = [os.path.splitext(action)[0] for action in os.listdir(dynamic_model_converted_data_path) if
               os.path.splitext(action)[1] == ".h5"]  # Obtiene las acciones posibles desde la ruta de datos

    # Inicializa el modelo de MediaPipe Hands
    with Hands() as hands_model:
        video = cv2.VideoCapture(id_camera)  # Inicia la captura de video desde la cámara

        while video.isOpened():
            _, frame = video.read()  # Lee un fotograma de la cámara

            # Realiza la detección con MediaPipe y extrae los puntos clave
            image, results = mediapipe_detection(frame, hands_model)
            keypoints = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[
                0].landmark]).flatten() if results.multi_hand_landmarks else np.zeros(
                21 * 3)  # Extrae los puntos clave de la primera mano detectada
            keypoint_sequence.append(keypoints)

            # Verifica si la secuencia de puntos clave tiene la longitud suficiente
            if len(keypoint_sequence) > max_length_frames and results.multi_hand_landmarks is not None:
                # Realiza la predicción del modelo con la secuencia de puntos clave
                res = model.predict(np.expand_dims(keypoint_sequence[-max_length_frames:], axis=0))[0]

                if res[np.argmax(res)] > threshold:
                    sent = actions[np.argmax(res)]
                    if sent == "nn":
                        sent = "ñ"
                    if len(sentence) == 0 or (len(sentence) > 0 and sentence[0] != sent):
                        sentence.insert(0, sent)  # Inserta la acción detectada en la oración
                        sentence = sentence[:10]  # Limita la longitud de la oración a 10 caracteres

                keypoint_sequence = []

            # Dibuja la interfaz gráfica
            height, width, _ = image.shape
            cv2.rectangle(image, (0, 0), (width, 50), (50, 50, 50), -1)  # Fondo gris oscuro
            display_sentence = ' '.join(sentence)
            cv2.putText(image, display_sentence,
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)  # Texto blanco

            if results.multi_hand_landmarks:  # Si se detectaron manos
                for hand_landmarks in results.multi_hand_landmarks:  # Itera sobre cada mano detectada
                    draw_landmarks(
                        image,
                        hand_landmarks,
                        HAND_CONNECTIONS,
                        # Especificaciones de dibujo para los puntos
                        DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        # Especificaciones de dibujo para las conexiones
                        DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
                    )

            cv2.imshow('Sign Language Translator', image)  # Muestra la imagen en una ventana
            if cv2.waitKey(10) & 0xFF == ord('q'):  # Permite salir del bucle presionando 'q'
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Se carga el modelo
    model_path = os.path.join(generated_models_path, dynamic_model_name)
    dynamic_model = load_model(model_path)  # Carga el modelo entrenado desde el archivo
    evaluate_dynamic_model(dynamic_model)  # Inicia la evaluación del modelo
