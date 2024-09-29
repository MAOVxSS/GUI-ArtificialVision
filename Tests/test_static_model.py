import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import mediapipe as mp

# Rutas y variables
from Utils.paths import static_model_data_labels_path, generated_models_path
from Utils.config import static_model_name, id_camera

# Cargar el archivo de etiquetas predefinido
labels_df = pd.read_csv(static_model_data_labels_path, header=None, index_col=0)
labels_dict = labels_df[1].to_dict()

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Cargar el modelo .keras
model_path = os.path.join(generated_models_path, static_model_name)
model = tf.keras.models.load_model(model_path)

def predict_keypoints(keypoints):
    input_data = np.array(keypoints, dtype=np.float32).reshape(1, 21 * 2)
    predictions = model.predict(input_data)
    return np.argmax(predictions), np.max(predictions)

def main():
    cap = cv2.VideoCapture(id_camera)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                keypoints = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks.landmark]).flatten()
                # Restar la posición de la muñeca a las coordenadas x e y
                wrist_x, wrist_y = keypoints[0], keypoints[1]
                keypoints -= [wrist_x, wrist_y] * 21
                predicted_label_id, confidence = predict_keypoints(keypoints)
                predicted_label = labels_dict[predicted_label_id]
                cv2.putText(frame, f'{predicted_label} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Keypoints', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
