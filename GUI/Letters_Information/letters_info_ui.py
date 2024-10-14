from pathlib import Path
from tkinter import Button, PhotoImage

# Funciones auxiliares
from Utils.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG
# Rutas
from Utils.paths import assets_letters_information_path, assets_dictionary_alphabet_path


# Ruta hacia el archivo JSON e imágenes de la seña
def relative_to_assets_data(path: str) -> Path:
    return assets_letters_information_path / Path(path)


# Ruta hacia los iconos
def relative_to_assets_icons(path: str) -> Path:
    return assets_dictionary_alphabet_path / Path(path)


def create_letters_information_window(window):
    # Configurar la ventana
    setup_window(window, background_color="#369FD6")

    # Crear el canvas común
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes
    images = {}

    canvas.create_rectangle(
        # Este rectangulo tendra la imagen extraida con la ruta del JSON "sign_path"
        609.0,
        28.0,
        1338.0,
        529.0,
        fill="#233D4D",
        outline="")

    canvas.create_rectangle(
        # Este rectangulo contendra el texto dinamico "description" del JSON data.json
        28.0,
        219.0,
        558.0,
        738.0,
        fill="#233D4D",
        outline="")

    canvas.create_rectangle(
        # Este rectangulo es solo decorativo para el texto
        44.0,
        43.0,
        320.0,
        191.0,
        fill="#F6AA1C",
        outline="")

    # Esta imagen, la ruta séra extraida a traves de la ruta "icon_path" del archivo JSON
    # Por ahora se le da un icono en una carpeta local
    image_image_1 = PhotoImage(file=relative_to_assets_data("image_1.png"))
    images["image_1"] = image_image_1
    canvas.create_image(438.0, 117.0, image=image_image_1)

    # Botones
    button_image_go_back = PhotoImage(file=relative_to_assets_data("button_1.png"))
    images["button_1"] = button_image_go_back
    button_go_back = Button(image=button_image_go_back, **BUTTON_COMMON_CONFIG)
    button_go_back.place(x=598.0, y=567.0, width=154.0, height=144.0)

    button_image_go_home = PhotoImage(file=relative_to_assets_data("button_2.png"))
    images["button_2"] = button_image_go_home
    button_go_home = Button(image=button_image_go_home, **BUTTON_COMMON_CONFIG)
    button_go_home.place(x=757.0, y=565.0, width=203.0, height=146.0)

    button_image_go_camera = PhotoImage(file=relative_to_assets_data("button_3.png"))
    images["button_3"] = button_image_go_camera
    button_go_camera = Button(image=button_image_go_camera, **BUTTON_COMMON_CONFIG)
    button_go_camera.place(x=963.0, y=565.0, width=378.0, height=146.0)

    # Lógica del botón "Regresar"
    from Utils.gui_utils import go_dictionary_alphabet_window
    button_go_back.config(command=lambda: go_dictionary_alphabet_window(window))

    # Lógica del botón "Inicio"
    from Utils.gui_utils import go_home_window
    button_go_home.config(command=lambda: go_home_window(window))

    window.resizable(False, False)

    return button_go_back, button_go_home, button_go_camera, images
