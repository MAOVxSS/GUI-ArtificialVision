import os
import numpy as np
import cv2
from mediapipe.python.solutions.hands import Hands
from mediapipe.python.solutions.hands import HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec

# Rutas y variables
from Utils.paths import dynamic_model_frames_data_path
from Utils.config import id_camera

# Función con la lógica para el guardado de frames de cada acción
def capture_samples(path, margin_frame=2, min_cant_frames=5):
    # Crea la carpeta de destino si no existe
    if not os.path.exists(path):
        os.makedirs(path)

    # Obtiene el siguiente índice de muestra basado en los subdirectorios 'sample_x' existentes
    existing_samples = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("sample_")]
    if not existing_samples:
        next_sample_index = 1
    else:
        existing_indices = [int(d.split("_")[1]) for d in existing_samples]
        next_sample_index = max(existing_indices) + 1

    frames = []

    # Inicializa el modelo de detección de manos de MediaPipe
    with Hands() as hands_model:
        video = cv2.VideoCapture(id_camera)  # Abre la cámara

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("Error: no se pudo leer el frame del video.")
                break

            # Realiza la detección de manos en el frame actual
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convierte la imagen de BGR a RGB
            image.flags.writeable = False  # Optimiza la velocidad de procesamiento
            results = hands_model.process(image)  # Procesa la imagen con el modelo de MediaPipe
            image.flags.writeable = True  # Vuelve a permitir escritura en la imagen
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convierte la imagen de vuelta a BGR

            # Dibuja los puntos clave de la mano en la imagen
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    draw_landmarks(
                        image,
                        hand_landmarks,
                        HAND_CONNECTIONS,
                        DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                    )

            # Muestra las instrucciones en la pantalla
            draw_text = 'Presiona "c" para capturar el frame, "e" para detener, "q" para salir'
            cv2.putText(image, draw_text, (5, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))

            # Muestra la imagen con las anotaciones en una ventana
            cv2.imshow(f'Acciones para la letra: "{os.path.basename(path)}"', image)

            # Captura las teclas presionadas para controlar la grabación
            key = cv2.waitKey(10) & 0xFF
            if key == ord('c'):
                # Captura el frame actual si se detecta una mano
                if results.multi_hand_landmarks:
                    print("Frame capturado.")
                    frames.append(np.asarray(frame))  # Guarda el frame actual
                else:
                    print("No se detectó una mano en el frame actual.")

            elif key == ord('e'):
                # Finaliza la captura de frames y guarda los samples
                if len(frames) > min_cant_frames:
                    print(f"Guardando {len(frames)} frames.")
                    output_folder = os.path.join(path, f"sample_{next_sample_index}")
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    # Guarda los frames en la carpeta especificada
                    for num_frame, frame in enumerate(frames):
                        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
                        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))
                    print(f"Frames guardados en: {output_folder}")
                    next_sample_index += 1
                else:
                    print("No se guardaron frames. Captura insuficiente.")

                # Reinicia los frames para la siguiente captura
                frames = []

            elif key == ord('q'):
                # Sale del bucle y cierra el programa
                break

        # Libera la cámara y destruye todas las ventanas de OpenCV
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    word_name = "z"  # Cambiar a la letra que se quiera guardar
    word_path = os.path.join(dynamic_model_frames_data_path, word_name)  # Ruta de guardado de los frames
    print(f"Guardando frames en: {word_path}")
    capture_samples(word_path)
