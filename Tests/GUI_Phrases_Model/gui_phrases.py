import os
import cv2
import json
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import joblib
import mediapipe as mp  # Importamos mediapipe
from keras.models import load_model
from Model.text_to_speech import initialize_tts, speak_text
from Utils.paths import phrases_model_json_data_path, phrases_model_keras_path
from Utils.config import phrases_model_keras_name, id_camera


class SignLanguageTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traductor de Lengua de Señas Mexicana")

        # Configuración de estilo minimalista
        self.root.configure(bg='#f0f0f0')

        # Título principal
        self.title_label = tk.Label(
            root,
            text="Traductor de Lengua de Señas",
            font=('Arial', 24, 'bold'),
            bg='#f0f0f0',
            fg='#333333'
        )
        self.title_label.pack(pady=(20, 10))

        # Frame principal
        self.main_frame = tk.Frame(root, bg='#f0f0f0')
        self.main_frame.pack(padx=20, pady=10, expand=True, fill=tk.BOTH)

        # Frame de video
        self.video_frame = tk.Label(self.main_frame, bg='black')
        self.video_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 10))

        # Frame de traducciones
        self.translation_frame = tk.Frame(self.main_frame, bg='white', borderwidth=2, relief=tk.RAISED)
        self.translation_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Etiqueta de traducciones
        self.translations_label = tk.Label(
            self.translation_frame,
            text="Traducción Actual:",
            font=('Arial', 20),
            bg='white',
            fg='#333333'
        )
        self.translations_label.pack(pady=(10, 5))

        # Área de texto de traducción
        self.translation_text = tk.Label(
            self.translation_frame,
            text="",
            font=('Arial', 18),
            bg='white',
            fg='#0066cc',
            wraplength=400
        )
        self.translation_text.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        # Área de porcentaje de confianza
        self.confidence_label = tk.Label(
            self.translation_frame,
            text="",
            font=('Arial', 16),
            bg='white',
            fg='#666666'
        )
        self.confidence_label.pack(pady=(0, 10))

        # Preparar modelo y configuraciones
        self.prepare_model()

        # Variables para detección
        self.kp_seq = []
        self.count_frame = 0
        self.fix_frames = 0
        self.recording = False

        # Iniciar captura de video
        self.video_capture = cv2.VideoCapture(id_camera)

        # Inicializar modelo Holistic de MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils  # Herramientas de dibujo
        self.mp_drawing_styles = mp.solutions.drawing_styles  # Estilos de dibujo
        self.mp_holistic = mp.solutions.holistic  # Modelo Holistic
        self.holistic_model = self.mp_holistic.Holistic()

        # Comenzar actualización de frames
        self.update_frame()

    def prepare_model(self):
        try:
            # Cargar el scaler
            scaler_path = os.path.join(phrases_model_keras_path, 'scaler.save')
            self.scaler = joblib.load(scaler_path)

            # Cargar identificadores de palabras desde el archivo JSON
            with open(phrases_model_json_data_path, 'r') as json_file:
                data = json.load(json_file)
                self.word_ids = data.get('word_ids')
                if not self.word_ids:
                    raise ValueError("[ERROR] No se encontraron identificadores de palabras en el archivo JSON.")

            # Cargar el modelo entrenado
            self.model = load_model(os.path.join(phrases_model_keras_path, phrases_model_keras_name))

            # Configuraciones de detección
            self.MODEL_FRAMES = 15
            self.MIN_LENGTH_FRAMES = 5
            self.threshold = 0.7
            self.margin_frame = 2
            self.delay_frames = 3

            # Diccionario de traducciones
            self.phrases_to_text = {
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
        except Exception as e:
            print(f"Error al preparar el modelo: {e}")
            raise

    def interpolate_keypoints(self, keypoints, target_length=15):
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
                interpolated_point = (1 - weight) * np.array(keypoints[lower_idx]) + weight * np.array(
                    keypoints[upper_idx])
                interpolated_keypoints.append(interpolated_point.tolist())

        return interpolated_keypoints

    def normalize_keypoints(self, keypoints, target_length=15):
        current_length = len(keypoints)
        if current_length < target_length:
            return self.interpolate_keypoints(keypoints, target_length)
        elif current_length > target_length:
            step = current_length / target_length
            indices = np.arange(0, current_length, step).astype(int)[:target_length]
            return [keypoints[i] for i in indices]
        else:
            return keypoints

    def update_frame(self):
        try:
            ret, frame = self.video_capture.read()
            if not ret:
                return

            # Procesar el frame con MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # Mejor rendimiento
            results = self.holistic_model.process(image)

            image.flags.writeable = True  # Habilitar modificaciones en la imagen

            # Dibujar los keypoints en la imagen
            self.mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            self.mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
            self.mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )

            # Detectar manos
            hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

            # Captura los frames mientras haya manos detectadas
            if hand_detected or self.recording:
                self.recording = False
                self.count_frame += 1
                if self.count_frame > self.margin_frame:
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
                    self.kp_seq.append(kp_frame)
            else:
                if self.count_frame >= self.MIN_LENGTH_FRAMES + self.margin_frame:
                    self.fix_frames += 1
                    if self.fix_frames < self.delay_frames:
                        self.recording = True
                        self.update_video_frame(image)
                        self.root.after(10, self.update_frame)
                        return

                    self.kp_seq = self.kp_seq[: -self.delay_frames] if self.delay_frames > 0 else self.kp_seq
                    kp_normalized = self.normalize_keypoints(self.kp_seq, int(self.MODEL_FRAMES))

                    kp_normalized = np.array(kp_normalized)
                    num_frames, num_keypoints = kp_normalized.shape

                    # Aplicar el scaler
                    kp_flat = kp_normalized.reshape(-1, num_keypoints)
                    kp_scaled = self.scaler.transform(kp_flat)
                    kp_scaled = kp_scaled.reshape(num_frames, num_keypoints)

                    res = self.model.predict(np.expand_dims(kp_scaled, axis=0))[0]

                    # Selecciona la palabra con mayor probabilidad si supera el umbral
                    if res[np.argmax(res)] > self.threshold:
                        word_id = self.word_ids[np.argmax(res)].split('-')[0]

                        # Obtiene la traducción limpia desde `phrases_to_text`
                        spoken_text = self.phrases_to_text.get(word_id, word_id)

                        # Actualizar traducción en interfaz
                        self.translation_text.config(text=self.phrases_to_text.get(word_id, word_id))

                        # Mostrar porcentaje de confianza
                        confidence = res[np.argmax(res)] * 100
                        self.confidence_label.config(text=f"Confianza: {confidence:.2f}%")

                self.recording = False
                self.fix_frames = 0
                self.count_frame = 0
                self.kp_seq = []

            # Convertir la imagen a BGR para mostrarla correctamente en Tkinter
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.update_video_frame(image_bgr)

        except Exception as e:
            print(f"Error en update_frame: {e}")

        # Programar siguiente actualización
        self.root.after(10, self.update_frame)

    def update_video_frame(self, frame):
        # Convertir frame de OpenCV a formato Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Redimensionar para ajustar
        img = img.resize((640, 480), Image.LANCZOS)

        # Convertir a PhotoImage
        imgtk = ImageTk.PhotoImage(image=img)

        # Actualizar label de video
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)


def main():
    root = tk.Tk()
    root.geometry("1200x700")  # Tamaño de ventana
    app = SignLanguageTranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
