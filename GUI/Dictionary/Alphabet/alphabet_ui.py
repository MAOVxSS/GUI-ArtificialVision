from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from pathlib import Path

# Rutas
from Utils.paths import assets_dictionary_alphabet_path
from Utils.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG

def relative_to_assets(path: str) -> Path:
    return assets_dictionary_alphabet_path / Path(path)

def create_dictionary_alphabet_window(window):
    # Configurar la ventana con los estilos comunes
    setup_window(window, background_color="#379FD7")

    # Crear el canvas común
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes
    images = {}

    # Botones
    button_image_a = PhotoImage(file=relative_to_assets("letter_a.png"))
    images["letter_a"] = button_image_a
    button_a = Button(image=button_image_a, **BUTTON_COMMON_CONFIG)
    button_a.place(x=172.0, y=62.0, width=127.0, height=114.0)

    button_image_u = PhotoImage(file=relative_to_assets("letter_u.png"))
    images["letter_u"] = button_image_u
    button_u = Button(image=button_image_u, **BUTTON_COMMON_CONFIG)
    button_u.place(x=245.0, y=455.0, width=127.0, height=114.0)

    button_image_g = PhotoImage(file=relative_to_assets("letter_g.png"))
    images["letter_g"] = button_image_g
    button_g = Button(image=button_image_g, **BUTTON_COMMON_CONFIG)
    button_g.place(x=1067.0, y=62.0, width=127.0, height=114.0)

    button_image_f = PhotoImage(file=relative_to_assets("letter_f.png"))
    images["letter_f"] = button_image_f
    button_f = Button(image=button_image_f, **BUTTON_COMMON_CONFIG)
    button_f.place(x=918.0, y=62.0, width=127.0, height=114.0)

    button_image_e = PhotoImage(file=relative_to_assets("letter_e.png"))
    images["letter_e"] = button_image_e
    button_e = Button(image=button_image_e, **BUTTON_COMMON_CONFIG)
    button_e.place(x=769.0, y=62.0, width=127.0, height=114.0)

    button_image_d = PhotoImage(file=relative_to_assets("letter_d.png"))
    images["letter_d"] = button_image_d
    button_d = Button(image=button_image_d, **BUTTON_COMMON_CONFIG)
    button_d.place(x=620.0, y=62.0, width=127.0, height=114.0)

    button_image_c = PhotoImage(file=relative_to_assets("letter_c.png"))
    images["letter_c"] = button_image_c
    button_c = Button(image=button_image_c, **BUTTON_COMMON_CONFIG)
    button_c.place(x=471.0, y=62.0, width=127.0, height=114.0)

    button_image_b = PhotoImage(file=relative_to_assets("letter_b.png"))
    images["letter_b"] = button_image_b
    button_b = Button(image=button_image_b, **BUTTON_COMMON_CONFIG)
    button_b.place(x=471.0, y=62.0, width=127.0, height=114.0)

    button_image_nn = PhotoImage(file=relative_to_assets("letter_nn.png"))
    images["letter_nn"] = button_image_nn
    button_nn = Button(image=button_image_nn, **BUTTON_COMMON_CONFIG)
    button_nn.place(x=172.0, y=318.0, width=127.0, height=114.0)











