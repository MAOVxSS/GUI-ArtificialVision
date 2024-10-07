from tkinter import Canvas, Button, PhotoImage
from pathlib import Path

# Rutas
from Utils.paths import assets_information_path
from Utils.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG

def relative_to_assets(path: str) -> Path:
    return assets_information_path / Path(path)

def create_information_window(window):
    # Configurar la ventana
    setup_window(window, background_color="#379FD7")

    # Crear el canvas común
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes
    images = {}

    # Colocar imágenes en el canvas
    image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
    images["image_1"] = image_image_1
    canvas.create_image(305.0, 91.0, image=image_image_1)

    image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
    images["image_2"] = image_image_2
    canvas.create_image(305.0, 460.0, image=image_image_2)

    image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
    images["image_3"] = image_image_3
    canvas.create_image(957.0, 321.0, image=image_image_3)

    # Crear botón "Regresar"
    button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
    images["button_1"] = button_image_1
    button_1 = Button(image=button_image_1, **BUTTON_COMMON_CONFIG)
    button_1.place(x=731.0, y=620.0, width=455.0, height=126.0)

    # Configurar la lógica del botón "Regresar"
    from Utils.gui_utils import on_back_button_home_click
    button_1.config(command=lambda: on_back_button_home_click(window))

    window.resizable(False, False)

    return button_1, images
