from tkinter import Tk, PhotoImage, Toplevel, Label
from PIL import Image, ImageTk
from Utils.gui_utils import center_window
import json
import cv2

# Rutas
DATA_FILE_PATH = "GUI/Assets/Letters_Information/data.json"
from pathlib import Path
from Utils.paths import assets_letters_information_path

# Variable global para la cámara
cap = None
video_update_id = None

# Ruta hacia el archivo JSON e imágenes de la seña
def relative_to_assets_data(path: str) -> Path:
    return assets_letters_information_path / Path(path)

# Función para generar la ventana "Inicio"
def go_camera_window(window, actual_letter):
    # Se guarda la letra actual para mantenerla en las ventanas
    actual_letter = actual_letter

    # Cargar el archivo JSON
    with open(DATA_FILE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if actual_letter in data:
        letter_data = data[actual_letter]

        # Funciones para crear la ventana
        from GUI.Camera.camera_ui import create_camera_window

        # Destruir la ventana actual (Information)
        window.destroy()

        # Crear una nueva ventana para Home
        new_window = Tk()
        center_window(new_window)

        # Crear la interfaz "home"
        button_tip, button_go_back, button_go_home, images = create_camera_window(new_window, actual_letter)

        update_icon_letter(new_window, letter_data["icon_path"])

        # Iniciar el loop principal para la nueva ventana
        new_window.mainloop()
    else:
        print(f"Letra '{actual_letter}' no encontrada en el archivo JSON")


# Actualiza el icono
def update_icon_letter(window, icon_path):
    canvas = window.children['!canvas']  # Obtén el canvas de la ventana

    # Actualizar el icono dinámicamente
    image_icon = PhotoImage(file=icon_path)
    canvas.create_image(260.0, 290.0, image=image_icon)

    # Mantener la referencia de la imagen para evitar que se elimine
    window.image_icon = image_icon

# Función para mostrar el tip en la ventana de la camara
def show_tip_window(actual_letter):
    # Cargar el archivo JSON
    with open(DATA_FILE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Verificar si la letra actual existe en el archivo JSON
    if actual_letter in data:
        letter_data = data[actual_letter]

        # Crear la ventana emergente
        tip_window = Toplevel()
        tip_window.title(f"Tip para la letra {actual_letter}")

        # Definir el tamaño de la ventana emergente
        window_width = 573
        window_height = 430
        screen_width = tip_window.winfo_screenwidth()
        screen_height = tip_window.winfo_screenheight()

        # Calcular la posición para centrar la ventana emergente
        position_right = int(screen_width / 2 - window_width / 2)
        position_down = int(screen_height / 2 - window_height / 2)
        tip_window.geometry(f"{window_width}x{window_height}+{position_right}+{position_down}")

        # Evitar errores con la ruta
        if 'GUI/Assets/Letters_Information' in letter_data["sign_path"]:
            sign_path = letter_data["sign_path"]  # Usar la ruta tal como está
        else:
            sign_path = relative_to_assets_data(letter_data["sign_path"])  # Ruta relativa

        # Si el recurso es un GIF, mostrar el GIF animado
        if sign_path.endswith(".gif"):
            show_gif_in_tip_window(tip_window, sign_path)
        else:
            # Mostrar una imagen PNG
            image_sign = PhotoImage(file=sign_path)
            label = Label(tip_window, image=image_sign)
            label.pack(padx=10, pady=10)

            # Mantener referencia de la imagen para evitar que se elimine
            tip_window.image_sign = image_sign
    else:
        print(f"Letra '{actual_letter}' no encontrada en el archivo JSON")


# Función para mostrar un GIF animado en la ventana emergente
def show_gif_in_tip_window(window, gif_path):
    # Evitar errores con la ruta
    if 'GUI/Assets/Letters_Information' not in gif_path:
        gif_path = relative_to_assets_data(gif_path)  # Ruta relativa

    # Cargar el GIF y dividir en frames
    gif_image = Image.open(gif_path)
    frames = []

    try:
        while True:
            frame = ImageTk.PhotoImage(gif_image.copy())
            frames.append(frame)
            gif_image.seek(len(frames))  # Avanzar al siguiente frame
    except EOFError:
        pass  # Termina cuando ya no hay más frames

    # Crear una etiqueta para mostrar el GIF
    label = Label(window)
    label.pack(padx=10, pady=10)

    # Definir la animación del GIF
    def update_gif(index):
        label.config(image=frames[index])
        window.after(100, update_gif, (index + 1) % len(frames))

    # Iniciar la animación del GIF
    update_gif(0)

    # Mantener referencia de los frames para evitar que se eliminen
    window.frames = frames

# Función para la captura de video
def start_video_stream(label):
    global cap, video_update_id
    cap = cv2.VideoCapture(1)  # Iniciar la captura de la cámara web (0 es la cámara por defecto)

    def update_frame():
        global video_update_id
        # Leer un frame de la cámara
        ret, frame = cap.read()
        if ret:
            # Convertir de BGR a RGB (OpenCV usa BGR por defecto)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convertir la imagen de OpenCV a un formato que Tkinter puede mostrar
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk  # Necesario para evitar que la imagen sea eliminada por el recolector de basura
            label.config(image=imgtk)

        # Volver a llamar a esta función después de 10ms
        video_update_id = label.after(10, update_frame)

    # Iniciar el bucle de actualización de frames
    update_frame()


def stop_video_stream(label):
    global cap, video_update_id
    if cap:
        cap.release()  # Liberar la cámara
        cap = None

    # Cancelar el ciclo 'after' si está activo
    if video_update_id is not None:
        label.after_cancel(video_update_id)
        video_update_id = None
