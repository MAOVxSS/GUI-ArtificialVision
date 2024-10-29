from tkinter import Button, PhotoImage
from pathlib import Path

# Importación de funciones auxiliares de la GUI
from GUI.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG
# Importación de la ruta hacia los recursos necesarios para la ventana
from Utils.paths import assets_dictionary_path

# Función para generar la ruta completa hacia los archivos de recursos (imágenes)
def relative_to_assets_dictionary(path: str) -> Path:
    return assets_dictionary_path / Path(path)

# Función para crear la ventana del diccionario
def create_dictionary_window(window):
    # Configurar la ventana con los estilos comunes definidos en 'setup_window'
    setup_window(window, background_color="#379FD7")

    # Crear un canvas común que actuará como contenedor de los elementos de la interfaz
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes para evitar que sean recolectadas por el recolector de basura
    images = {}

    # Imagen decorativa en el lado izquierdo del canvas
    image_image_1 = PhotoImage(file=relative_to_assets_dictionary("image_1.png"))
    images["image_1"] = image_image_1
    canvas.create_image(339.0, 384.0, image=image_image_1)  # Posición de la imagen en el canvas

    # Botón para acceder al alfabeto del diccionario
    button_image_alphabet = PhotoImage(file=relative_to_assets_dictionary("alfabeto.png"))
    images["alfabeto"] = button_image_alphabet
    button_alphabet = Button(image=button_image_alphabet, **BUTTON_COMMON_CONFIG)
    button_alphabet.place(x=704.0, y=250.0, width=546.0, height=132.0)

    # Botón para acceder al vocabulario del diccionario (a implementar)
    button_image_vocabulary = PhotoImage(file=relative_to_assets_dictionary("vocabulario.png"))
    images["vocabulario"] = button_image_vocabulary
    button_vocabulary = Button(image=button_image_vocabulary, **BUTTON_COMMON_CONFIG)
    button_vocabulary.place(x=741.0, y=405.0, width=457.0, height=123.0)

    # Imagen decorativa en la parte superior del canvas
    image_image_2 = PhotoImage(file=relative_to_assets_dictionary("image_2.png"))
    images["image_2"] = image_image_2
    canvas.create_image(981.0, 136.0, image=image_image_2)  # Posición de la imagen en el canvas

    # Botón para regresar a la ventana principal (home)
    button_image_go_back = PhotoImage(file=relative_to_assets_dictionary("regresar.png"))
    images["regresar"] = button_image_go_back
    button_go_back = Button(image=button_image_go_back, **BUTTON_COMMON_CONFIG)
    button_go_back.place(x=741.0, y=555.0, width=450.0, height=124.0)

    # Importar funciones necesarias para la navegación entre ventanas
    from GUI.gui_utils import go_home_window
    from GUI.gui_utils import go_dictionary_alphabet_window

    # Lógica del botón "Regresar" para volver a la ventana principal (home)
    button_go_back.config(command=lambda: go_home_window(window))

    # Lógica del botón "Alfabeto" para acceder a la ventana del alfabeto del diccionario
    button_alphabet.config(command=lambda: go_dictionary_alphabet_window(window))

    # Desactivar la opción de redimensionar la ventana (para mantener un diseño fijo)
    window.resizable(False, False)

    # Retornar los botones y las referencias a las imágenes (para gestión adicional si se requiere)
    return button_alphabet, button_vocabulary, button_go_back, images
