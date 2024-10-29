from tkinter import Button, PhotoImage
from PIL import Image, ImageTk
from pathlib import Path
from GUI.Home.home_logic import show_tese_gif, stop_gif_animation

# Importación de funciones auxiliares de la GUI
from GUI.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG
# Importación de la ruta hacia los recursos necesarios para la ventana
from Utils.paths import assets_home_path


# Función para generar la ruta completa hacia los archivos de recursos (imágenes y GIF)
def relative_to_assets_home(path: str) -> Path:
    return assets_home_path / Path(path)


# Función para crear la ventana principal (Home)
def create_home_window(window):
    # Configurar la ventana principal con un color de fondo
    setup_window(window, background_color="#369FD6")

    # Crear un canvas común que actuará como contenedor de los elementos de la interfaz
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes para evitar que sean recolectadas por el recolector de basura
    images = {}

    # Crear botones e imágenes
    # Botón para acceder a la ventana "Diccionario"
    button_image_dictionary = PhotoImage(file=relative_to_assets_home("diccionario.png"))
    images["diccionario"] = button_image_dictionary
    button_dictionary = Button(image=button_image_dictionary, **BUTTON_COMMON_CONFIG)
    button_dictionary.place(x=88.0, y=478.0, width=446.0, height=130.0)

    # Botón para acceder a la ventana "Información" (ícono de información)
    button_image_information = PhotoImage(file=relative_to_assets_home("informacion.png"))
    images["informacion"] = button_image_information
    button_information = Button(image=button_image_information, **BUTTON_COMMON_CONFIG)
    button_information.place(x=1224.0, y=629.0, width=115.0, height=121.0)

    # Botón para acceder a la ventana "Lecciones"
    button_image_lessons = PhotoImage(file=relative_to_assets_home("lecciones.png"))
    images["lecciones"] = button_image_lessons
    button_lessons = Button(image=button_image_lessons, **BUTTON_COMMON_CONFIG)
    button_lessons.place(x=858.0, y=486.0, width=446.0, height=130.0)

    # Colocar imágenes decorativas en el canvas
    image_image_1 = PhotoImage(file=relative_to_assets_home("image_1.png"))
    images["image_1"] = image_image_1
    canvas.create_image(683.0, 270.0, image=image_image_1)

    image_image_2 = PhotoImage(file=relative_to_assets_home("image_2.png"))
    images["image_2"] = image_image_2
    canvas.create_image(239.0, 97.0, image=image_image_2)

    image_image_3 = PhotoImage(file=relative_to_assets_home("image_3.png"))
    images["image_3"] = image_image_3
    canvas.create_image(1052.0, 99.0, image=image_image_3)

    image_image_4 = PhotoImage(file=relative_to_assets_home("image_4.png"))
    images["image_4"] = image_image_4
    canvas.create_image(1030.0, 704.0, image=image_image_4)

    # Mostrar el GIF
    show_tese_gif(window, canvas)

    # Importar las funciones para crear las ventanas
    from GUI.gui_utils import go_dictionary_window, go_lessons_window, go_information_window

    # Asignar la función correspondiente a cada botón
    button_dictionary.config(command=lambda: (stop_gif_animation(window), go_dictionary_window(window)))
    button_lessons.config(command=lambda: (stop_gif_animation(window), go_lessons_window(window)))
    button_information.config(command=lambda: (stop_gif_animation(window), go_information_window(window)))

    # Desactivar la opción de redimensionar la ventana (para mantener un diseño fijo)
    window.resizable(False, False)

    # Retornar los botones y las referencias a las imágenes
    return button_dictionary, button_information, button_lessons, images
