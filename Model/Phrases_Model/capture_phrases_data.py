import os
import cv2
import numpy as np
from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic
from Model.model_utils import draw_keypoints
from Utils.paths import phrases_model_frames_data_path, phrases_model_test_frames_data_path
from Utils.config import id_camera

def capture_samples(path, margin_frame=2, min_cant_frames=5, delay_frames=3):
    """
    Captura de muestras para una acción guardando secuencia de frames

    Parámetros:
    path -- Ruta donde se guardarán los frames capturados de cada muestra de palabra.
    margin_frame -- Número de frames ignorados al comienzo y al final para estabilizar la captura.
    min_cant_frames -- Mínimo número de frames necesarios para guardar una muestra válida.
    delay_frames -- Número de frames extra capturados tras perder la detección de manos, evitando cortes bruscos.

    Funcionalidad:
     Captura secuencias de frames para una palabra o acción específica. Utiliza el modelo Holistic de MediaPipe
    para detectar manos y cuerpo en tiempo real. Los frames capturados se guardan en carpetas únicas basadas en
    un identificador temporal. La función asegura estabilidad descartando frames no confiables al inicio o al final.
    """

    # Crea el directorio de destino si no existe
    if not os.path.exists(path):
        os.makedirs(path)

    # Variables para controlar el conteo de frames y el estado de grabación
    count_frame = 0
    frames = []
    fix_frames = 0
    recording = False  # Indica si la captura de frames está activa

    # Inicializa el modelo Holistic de MediaPipe para detección de manos y cuerpo
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(id_camera)  # Abre la cámara con el ID especificado

        while video.isOpened():
            # Leer un frame de la cámara
            ret, frame = video.read()
            if not ret:
                break  # Si no se lee el frame, se termina el bucle

            # Procesar la imagen con MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
            image.flags.writeable = False  # Optimizar para el procesamiento
            results = holistic_model.process(image)  # Procesa la imagen con el modelo de MediaPipe
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convierte la imagen de vuelta a BGR

            # Verifica si se detecta una mano en los resultados
            hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

            # Si se detecta una mano o si ya se está grabando, continúa la captura de frames
            if hand_detected or recording:
                recording = False  # Reset de grabación a False mientras detecta la mano
                count_frame += 1  # Incrementa el contador de frames capturados

                # Al sobrepasar el margen inicial, empieza a guardar los frames capturados
                if count_frame > margin_frame:
                    cv2.putText(
                        image,
                        'Capturando...',
                        (5, 30),
                        cv2.FONT_ITALIC,
                        2,
                        (255, 50, 0))
                    frames.append(np.asarray(frame))  # Agrega el frame actual a la lista de frames

            # Si no se detecta una mano y ya se ha capturado la cantidad mínima de frames
            else:
                if len(frames) >= min_cant_frames + margin_frame:
                    fix_frames += 1  # Aumenta el contador de delay

                    # Si se alcanza el delay antes de detener, sigue grabando unos frames extra
                    if fix_frames < delay_frames:
                        recording = True
                        continue

                    # Descarta frames finales para evitar falsos positivos al salir de la seña
                    frames = frames[: - (margin_frame + delay_frames)]

                    # Crea una carpeta para la muestra actual basada en la fecha y hora
                    actual_datatime = datetime.now().strftime('%y%m%d%H%M%S%f')
                    output_folder = os.path.join(path, f"sample_{actual_datatime}")
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    # Guarda cada frame en el directorio de salida
                    for num_frame, frame in enumerate(frames):
                        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
                        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))  # Guardar en BGRA

                # Reinicia variables para la próxima captura de seña
                recording, fix_frames = False, 0
                frames, count_frame = [], 0
                cv2.putText(
                    image,
                    'Listo para capturar...',
                    (5, 30),
                    cv2.FONT_ITALIC,
                    2,
                    (0, 220, 100))
            # Dibujar los puntos clave en la imagen
            draw_keypoints(image, results)

            # Mostrar la imagen procesada en la ventana
            cv2.imshow(f'Muestras para: "{os.path.basename(path)}"', image)

            # Si se presiona la tecla 'q', sale del bucle y cierra la ventana
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Libera la cámara y destruye todas las ventanas de OpenCV al finalizar
        video.release()
        cv2.destroyAllWindows()

# Configuración inicial para especificar la palabra a capturar y la ubicación de guardado
if __name__ == "__main__":
    word_name = "como_estas"  # Cambiar a la palabra o frase que se desea capturar y se agrega "_test" si son de prueba

    # Guardado normal
    word_path = os.path.join(phrases_model_frames_data_path, word_name)

    # Guardado de prueba
    # word_path = os.path.join(phrases_model_test_frames_data_path, word_name)

    capture_samples(word_path)  # Iniciar captura de muestras
