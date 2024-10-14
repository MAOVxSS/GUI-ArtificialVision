from tkinter import Canvas, Button, PhotoImage
from pathlib import Path

# Rutas
from Utils.paths import assets_dictionary_path
from Utils.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG

def relative_to_assets(path: str) -> Path:
    return assets_dictionary_path / Path(path)

def create_lessons_window(window):
    # Configurar la ventana con los estilos comunes
    setup_window(window, background_color="#379FD7")

    # Crear el canvas común
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes
    images = {}

    image_image_1 = PhotoImage(file=relative_to_assets("avatar.png"))
    images["avatar"] = image_image_1
    canvas.create_image(317.0, 415.0, image=image_image_1)

    button_image_alphabet = PhotoImage(file=relative_to_assets("alfabeto.png"))
    images["alfabeto"] = button_image_alphabet
    button_alphabet = Button(image=button_image_alphabet, **BUTTON_COMMON_CONFIG)
    button_alphabet.place(x=704.0, y=250.0, width=546.0, height=132.0)

    button_image_vocabulary = PhotoImage(file=relative_to_assets("vocabulario.png"))
    images["vocabulario"] = button_image_vocabulary
    button_vocabulary = Button(image=button_image_vocabulary, **BUTTON_COMMON_CONFIG)
    button_vocabulary.place(x=741.0, y=405.0, width=457.0, height=123.0)

    image_image_2 = PhotoImage(file=relative_to_assets("lecciones_titulo.png"))
    images["lecciones_titulo"] = image_image_2
    canvas.create_image(981.0, 136.0, image=image_image_2)

    button_image_go_back = PhotoImage(file=relative_to_assets("regresar.png"))
    images["regresar"] = button_image_go_back
    button_go_back = Button(image=button_image_go_back, **BUTTON_COMMON_CONFIG)
    button_go_back.place(x=741.0, y=555.0, width=450.0, height=124.0)

    # Lógica del botón "Regresar"
    from Utils.gui_utils import go_home_window
    button_go_back.config(command=lambda: go_home_window(window))

    # Lógica del botón "Alfabeto"
    from  Utils.gui_utils import go_lessons_alphabet_window
    button_alphabet.config(command=lambda: go_lessons_alphabet_window(window))

    # button_alphabet.config(command=lambda:)

    window.resizable(False, False)

    return button_alphabet, button_vocabulary, button_go_back, images
