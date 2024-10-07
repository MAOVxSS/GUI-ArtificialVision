from tkinter import Canvas, Button, PhotoImage
from pathlib import Path

# Rutas
from Utils.paths import assets_dictionary_path
from Utils.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG

def relative_to_assets(path: str) -> Path:
    return assets_dictionary_path / Path(path)

def create_dictionary_window(window):
    # Configurar la ventana con los estilos comunes
    setup_window(window, background_color="#379FD7")

    # Crear el canvas común
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes
    images = {}

    # Colocar imágenes y botones en el canvas
    image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
    images["image_1"] = image_image_1
    canvas.create_image(339.0, 384.0, image=image_image_1)

    button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
    images["button_1"] = button_image_1
    button_1 = Button(image=button_image_1, **BUTTON_COMMON_CONFIG, command=lambda: print("button_1 clicked"))
    button_1.place(x=704.0, y=250.0, width=546.0, height=132.0)

    button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
    images["button_2"] = button_image_2
    button_2 = Button(image=button_image_2, **BUTTON_COMMON_CONFIG, command=lambda: print("button_2 clicked"))
    button_2.place(x=741.0, y=405.0, width=457.0, height=123.0)

    image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
    images["image_2"] = image_image_2
    canvas.create_image(981.0, 136.0, image=image_image_2)

    button_image_3 = PhotoImage(file=relative_to_assets("button_3.png"))
    images["button_3"] = button_image_3
    button_3 = Button(image=button_image_3, **BUTTON_COMMON_CONFIG, command=lambda: print("button_3 clicked"))
    button_3.place(x=741.0, y=555.0, width=450.0, height=124.0)

    # Configurar la lógica del botón "Regresar"
    from Utils.gui_utils import on_back_button_home_click
    button_3.config(command=lambda: on_back_button_home_click(window))

    window.resizable(False, False)

    return button_1, button_2, button_3, images
