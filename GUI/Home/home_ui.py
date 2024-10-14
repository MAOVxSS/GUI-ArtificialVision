from tkinter import Tk, Button, PhotoImage
from PIL import Image, ImageTk
from pathlib import Path

# Rutas
from Utils.paths import assets_home_path

# Funciones auxiliares
from Utils.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG

# Variables globales para gestionar el GIF
gif_running = False
gif_animation_id = None

def relative_to_assets(path: str) -> Path:
    return assets_home_path / Path(path)

def create_home_window(window):
    global gif_running, gif_animation_id
    gif_running = False  # Detenemos cualquier animación de GIF anterior

    # Cancelar el ciclo `after` si hay uno activo
    if gif_animation_id is not None:
        try:
            window.after_cancel(gif_animation_id)
        except ValueError:
            pass  # Si el ID no es válido, no hay necesidad de cancelar

    # Configurar la ventana
    setup_window(window, background_color="#369FD6")

    # Crear el canvas común
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes
    images = {}

    # Crear botones e imágenes
    button_image_dictionary = PhotoImage(file=relative_to_assets("diccionario.png"))
    images["diccionario"] = button_image_dictionary
    button_dictionary = Button(image=button_image_dictionary, **BUTTON_COMMON_CONFIG)
    button_dictionary.place(x=88.0, y=478.0, width=446.0, height=130.0)

    button_image_information = PhotoImage(file=relative_to_assets("informacion.png"))
    images["informacion"] = button_image_information
    button_information = Button(image=button_image_information, **BUTTON_COMMON_CONFIG)
    button_information.place(x=1224.0, y=629.0, width=115.0, height=121.0)

    button_image_lessons = PhotoImage(file=relative_to_assets("lecciones.png"))
    images["lecciones"] = button_image_lessons
    button_lessons = Button(image=button_image_lessons, **BUTTON_COMMON_CONFIG)
    button_lessons.place(x=858.0, y=486.0, width=446.0, height=130.0)

    # Colocar imágenes en el canvas
    image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
    images["image_1"] = image_image_1
    canvas.create_image(683.0, 270.0, image=image_image_1)

    image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
    images["image_2"] = image_image_2
    canvas.create_image(239.0, 97.0, image=image_image_2)

    image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
    images["image_3"] = image_image_3
    canvas.create_image(1052.0, 99.0, image=image_image_3)

    image_image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
    images["image_4"] = image_image_4
    canvas.create_image(1030.0, 704.0, image=image_image_4)

    # Cargar y mostrar el GIF en lugar del rectángulo
    gif_path = relative_to_assets("logo_tese_animado.gif")
    gif_image = Image.open(gif_path)

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
            canvas.itemconfig(gif_item, image=frames[index])
            gif_animation_id = window.after(100, update_gif, (index + 1) % len(frames))

    gif_item = canvas.create_image(647.0, 104.5, image=frames[0])
    gif_running = True  # Iniciar la animación
    update_gif(0)

    window.resizable(False, False)

    return button_dictionary, button_information, button_lessons, images

def stop_gif_animation(window):
    global gif_running, gif_animation_id
    gif_running = False  # Detener la animación del GIF
    if gif_animation_id is not None:
        try:
            # Cancelar cualquier tarea `after` activa
            window.after_cancel(gif_animation_id)
        except ValueError:
            pass
