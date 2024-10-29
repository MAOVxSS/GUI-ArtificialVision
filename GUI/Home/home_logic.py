from tkinter import Tk
from PIL import Image, ImageTk

# Función para centrar la ventana
from GUI.gui_utils import center_window

# Variables globales para gestionar la animación del GIF en la ventana de inicio
gif_running = False  # Indica si el GIF está en ejecución
gif_animation_id = None  # ID del ciclo `after` para animar el GIF


def show_tese_gif(window, canvas):
    global gif_running, gif_animation_id

    gif_running = False  # Detenemos cualquier animación de GIF anterior

    # Cancelar el ciclo `after` si hay uno activo
    if gif_animation_id is not None:
        try:
            window.after_cancel(gif_animation_id)
        except ValueError:
            pass  # Si el ID no es válido, no hay necesidad de cancelar

    # Cargar y mostrar el GIF en lugar del rectángulo
    gif_path = "GUI/Assets/Home/logo_tese_animado.gif"
    gif_image = Image.open(gif_path)

    # Extraer los frames del GIF y almacenarlos en una lista
    frames = []
    try:
        while True:
            frame = ImageTk.PhotoImage(gif_image.copy())
            frames.append(frame)
            gif_image.seek(len(frames))  # Avanzar al siguiente frame
    except EOFError:
        pass  # Termina cuando ya no hay más frames

        # Definir la animación del GIF
        def update_gif(index):
            global gif_animation_id
            if gif_running:  # Solo continuar si la animación debe seguir corriendo
                canvas.itemconfig(gif_item, image=frames[index])  # Actualizar el frame del GIF en el canvas
                # Programar la actualización del siguiente frame después de 100 ms
                gif_animation_id = window.after(100, update_gif, (index + 1) % len(frames))

        # Crear la imagen del GIF en el canvas, ubicada en la posición especificada
        gif_item = canvas.create_image(647.0, 104.5, image=frames[0])
        gif_running = True  # Iniciar la animación del GIF
        update_gif(0)  # Iniciar la actualización de los frames

# Función para detener la animación del GIF
def stop_gif_animation(window):
    global gif_running, gif_animation_id
    gif_running = False  # Detener la animación del GIF
    if gif_animation_id is not None:
        try:
            # Cancelar cualquier tarea `after` activa
            window.after_cancel(gif_animation_id)
        except ValueError:
            pass  # Si ya fue cancelada o no es válida, se ignora el error