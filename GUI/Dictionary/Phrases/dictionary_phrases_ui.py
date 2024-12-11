from pathlib import Path
from tkinter import Button, PhotoImage

# Importación de funciones auxiliares de la GUI
from GUI.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG

# Importación de la ruta hacia los recursos necesarios para la ventana
from Utils.paths import assets_dictionary_phrases_path

def relative_to_assets_dictionary_phrases(path: str) -> Path:
    return assets_dictionary_phrases_path / Path(path)

# Función para crear la ventana donde se muestran las frases disponibles
def create_dictionary_phrases_window(window):
    # Configurar la ventana principal con un color de fondo
    setup_window(window, background_color="#379FD7")

    # Crear un canvas común que actuará como contenedor de los elementos de la interfaz
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes para evitar que sean recolectadas por el recolector de basura
    images = {}

    # Botón para la frase "Hola"
    button_image_phrase_hello = PhotoImage(file=relative_to_assets_dictionary_phrases("frase_hola.png"))
    images["frase_hola"] = button_image_phrase_hello
    button_phrase_hello = Button(image=button_image_phrase_hello, **BUTTON_COMMON_CONFIG)
    button_phrase_hello.place(x=317.0, y=40.0, width=345.0, height=123.0)

    # Botón para la frase "Adios"
    button_image_phrase_bye = PhotoImage(file=relative_to_assets_dictionary_phrases("frase_adios.png"))
    images["frase_adios"] = button_image_phrase_bye
    button_phrase_bye = Button(image=button_image_phrase_bye, **BUTTON_COMMON_CONFIG)
    button_phrase_bye.place(x=699.0, y=40.0, width=359.0, height=123.0)

    # Botón para la frase "¿Cómo estás?"
    button_image_phrase_how_are_you = PhotoImage(file=relative_to_assets_dictionary_phrases("frase_como_estas.png"))
    images["frase_como_estas"] = button_image_phrase_how_are_you
    button_phrase_how_are_you = Button(image=button_image_phrase_how_are_you, **BUTTON_COMMON_CONFIG)
    button_phrase_how_are_you.place(x=317.0, y=164.0, width=345.0, height=123.0)

    # Botón para la frase "Cuídate"
    button_image_phrase_take_care = PhotoImage(file=relative_to_assets_dictionary_phrases("frase_cuidate.png"))
    images["frase_cuidate"] = button_image_phrase_take_care
    button_phrase_take_care = Button(image=button_image_phrase_take_care, **BUTTON_COMMON_CONFIG)
    button_phrase_take_care.place(x=699.0, y=164.0, width=359.0, height=123.0)

    # Botón para la frase "Más o menos"
    button_image_phrase_neutral = PhotoImage(file=relative_to_assets_dictionary_phrases("frase_mas_o_menos.png"))
    images["frase_mas_o_menos"] = button_image_phrase_neutral
    button_phrase_neutral = Button(image=button_image_phrase_neutral, **BUTTON_COMMON_CONFIG)
    button_phrase_neutral.place(x=699.0, y=287.0, width=359.0, height=123.0)

    # Botón para la frase "gracias"
    button_image_phrase_thank_you = PhotoImage(file=relative_to_assets_dictionary_phrases("frase_gracias.png"))
    images["frase_gracias"] = button_image_phrase_thank_you
    button_phrase_thank_you = Button(image=button_image_phrase_thank_you, **BUTTON_COMMON_CONFIG)
    button_phrase_thank_you.place(x=699.0, y=408.0, width=359.0, height=123.0)

    # Botón para la frase "por favor"
    button_image_phrase_please = PhotoImage(file=relative_to_assets_dictionary_phrases("frase_por_favor.png"))
    images["frase_por_favor"] = button_image_phrase_please
    button_phrase_please = Button(image=button_image_phrase_please, **BUTTON_COMMON_CONFIG)
    button_phrase_please.place(x=308.0, y=287.0, width=358.0, height=123.0)

    # Botón para la frase "de nada"
    button_image_phrase_you_are_welcome = PhotoImage(file=relative_to_assets_dictionary_phrases("frase_de_nada.png"))
    images["frase_de_nada"] = button_image_phrase_you_are_welcome
    button_phrase_you_are_welcome = Button(image=button_image_phrase_you_are_welcome, **BUTTON_COMMON_CONFIG)
    button_phrase_you_are_welcome.place(x=308.0, y=408.0, width=358.0, height=123.0)

    # Botón "Regresar" para volver a la ventana del diccionario
    button_image_go_back = PhotoImage(file=relative_to_assets_dictionary_phrases("regresar.png"))
    images["regresar"] = button_image_go_back
    button_go_back = Button(image=button_image_go_back, **BUTTON_COMMON_CONFIG)
    button_go_back.place(x=171.0, y=559.0, width=491.0, height=146.0)

    # Botón "Inicio" para volver a la ventana principal (home)
    button_image_go_home = PhotoImage(file=relative_to_assets_dictionary_phrases("inicio.png"))
    images["inicio"] = button_image_go_home
    button_go_home = Button(image=button_image_go_home, **BUTTON_COMMON_CONFIG)
    button_go_home.place(x=699.0, y=559.0, width=408.0, height=146.0)

    # Lógica para los botones de las frases
    from GUI.Phrases_Info.phrases_info_logic import button_phrase_click

    button_phrase_hello.config(command=lambda: button_phrase_click(window, "HOLA"))
    button_phrase_bye.config(command=lambda: button_phrase_click(window, "ADIOS"))
    button_phrase_how_are_you.config(command=lambda: button_phrase_click(window, "COMO_ESTAS"))
    button_phrase_take_care.config(command=lambda: button_phrase_click(window, "CUIDATE"))
    button_phrase_neutral.config(command=lambda: button_phrase_click(window, "MAS_O_MENOS"))
    button_phrase_thank_you.config(command=lambda: button_phrase_click(window, "GRACIAS"))
    button_phrase_please.config(command=lambda: button_phrase_click(window, "POR_FAVOR"))
    button_phrase_you_are_welcome.config(command=lambda: button_phrase_click(window, "DE_NADA"))


    # Lógica del botón "Regresar"
    from GUI.gui_utils import go_dictionary_window
    button_go_back.config(command=lambda: go_dictionary_window(window))

    # Lógica del botón "Inicio"
    from GUI.gui_utils import go_home_window
    button_go_home.config(command=lambda: go_home_window(window))

    # Desactivar la opción de redimensionar la ventana (para mantener un diseño fijo)
    window.resizable(False, False)

    # Retornar los botones y las referencias a las imágenes
    return (button_phrase_hello, button_phrase_bye, button_phrase_how_are_you, button_phrase_take_care,
            button_phrase_neutral, button_phrase_thank_you, button_phrase_please, button_phrase_you_are_welcome,
            button_go_back, button_go_home, images)